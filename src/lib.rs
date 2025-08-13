//! [Byte-Pair Encoding] (BPE) implementation from scratch, designed
//! specifically for tokenization tasks in NLP and text processing.
//!
//! [Byte-Pair Encoding]: https://en.wikipedia.org/wiki/Byte-pair_encoding

#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]

mod trie;

use std::collections::{HashMap, VecDeque};
use std::fmt::Write;
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::Path;
use trie::{FlattenTrie, Trie};

/// The size of a token ID for the `Tokenizer`.
///
/// Allows for up to ~65K unique token sequences.
#[allow(non_camel_case_types)]
pub type token_id_t = u16;

/// The maximum vocabulary size, considering token ID size, initial single-byte
/// tokens, and special tokens.
const MAX_VOCAB_SIZE: token_id_t = token_id_t::MAX - (256 + SPECIAL_TOKENS.len()) as token_id_t;

/// Used to provide structure and additional context for the model's learning.
const SPECIAL_TOKENS: [&str; 1] = ["<|eos|>"];

/// A byte-level [BPE] tokenizer, used to build a vocabulary of subword units
/// from a raw corpus.
///
/// Instead of fixed word/character tokens, BPE learns frequently co-occurring
/// sequences of bytes/characters from the data. This enables the tokenizer to
/// handle rare, unseen, or unknown characters by representing text as
/// combinations of subwords.
///
/// [BPE]: https://en.wikipedia.org/wiki/Byte-pair_encoding
///
/// # Example
///
/// ```no_run
/// use bytepair::{token_id_t, Tokenizer};
///
/// fn main() -> std::io::Result<()> {
///     let raw_text = std::fs::read_to_string("the_verdict.txt")?;
///
///     // Option 1: Load the tokenizer from a file containing merge rules.
///     let tokenizer = Tokenizer::load("merges.txt")?;
///
///     // Option 2: Build a tokenizer directly from the raw text.
///     // let tokenizer = Tokenizer::new(&raw_text)?;
///
///     // Encode the raw text into token IDs.
///     let encoded: Vec<token_id_t> = tokenizer.encode(&raw_text);
///
///     // Decode the token IDs back into it's string representation.
///     let decoded: String = tokenizer.decode(&encoded);
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct Tokenizer {
    /// Stores the vocabulary in a [FlattenTrie] for efficient text encoding.
    encoder: FlattenTrie,
    /// Maps token IDs to their corresponding token sequences.
    decoder: Vec<Vec<token_id_t>>,
}

impl Tokenizer {
    /// Creates a new `Tokenizer`, training it from the provided corpus.
    ///
    /// This function performs BPE training and saves the resulting merge rules
    /// to disk. For loading pre-existing merge rules, use [Tokenizer::load].
    ///
    /// # Panics
    ///
    /// Panics if the provided corpus is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the merge rules cannot be serialized to disk.
    pub fn new(corpus: &str) -> io::Result<Self> {
        // Used for efficient encoding by greedily matching the longest prefix
        // of the given input.
        let mut encoder: Trie = Default::default();

        // Used for efficient decoding, allowing for the indexing of token
        // sequences via token ID.
        let mut decoder: Vec<Vec<token_id_t>> = Default::default();

        // Initialize the encoder and decoder with all single-byte tokens
        // (0x00..=0xFF).
        //
        // Starting with single-byte tokens ensures the encoder can tokenize
        // any UTF-8 string, even characters not present in the training
        // corpus. Subsequent merges will build larger tokens from these base
        // units.
        for i in 0x00..=0xFF {
            let token = i as token_id_t;

            // Can be inserted as is.
            encoder.insert(&[token], token);
            decoder.push(vec![token]);
        }

        // The training corpus is kept as a flat token vector to allow BPE to
        // learn merges across whitespace and word boundaries, improving
        // tokenization and compression.
        //
        // BPE merges frequent byte sequences into subwords, so each element
        // must also represent both base bytes and token IDs.
        let mut training_corpus: Vec<_> = corpus
            .as_bytes()
            .iter()
            .map(|byte| *byte as token_id_t)
            .collect();

        // Merge rules are sequences of instructions produced during the BPE
        // training process. Each rule describes how two existing tokens are
        // combined into a new, larger token. They capture the order and
        // content of merges, allowing the vocabulary to be rebuilt from
        // scratch.
        let mut merge_rules: VecDeque<(token_id_t, token_id_t)> = Default::default();

        // NOTE: This assumes a token ID size of u16.
        let mut freq_map: HashMap<u32, u32> = Default::default();

        loop {
            // Finding the most frequent token pairs to merge reduces the token
            // count, compresses the vocabulary, and improves encoding/decoding
            // efficiency.
            count_pair_freqs(&training_corpus, &mut freq_map);

            // TODO: Could probably be improved.
            let (pair, count) = freq_map
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .expect("provided corpus is empty");

            // Stop if no more tokens can be merged or the vocabulary is full.
            if *count < 2 || decoder.len() >= MAX_VOCAB_SIZE as usize {
                break;
            }

            // Unpack the token pair.
            //
            // NOTE: This assumes a token ID size of u16.
            let tok_1 = (pair >> 16) as token_id_t;
            let tok_2 = (pair & 0xFFFF) as token_id_t;

            // Update the merge rules.
            merge_rules.push_back((tok_1, tok_2));

            // The new ID is based on the decoder length, incrementing from 256
            // after the initial single-byte tokens are inserted.
            let new_token_id = decoder.len() as token_id_t;

            // Since tokens can themselves be nested merges, the fully expanded
            // token sequences of the token pair should be inserted into the
            // encoder, so it can greedily match the longest possible sequences
            // when encoding new text.
            let mut merged_seq =
                Vec::with_capacity(decoder[tok_1 as usize].len() + decoder[tok_2 as usize].len());

            merged_seq.extend_from_slice(&decoder[tok_1 as usize]);
            merged_seq.extend_from_slice(&decoder[tok_2 as usize]);

            decoder.push(merged_seq);
            encoder.insert(&decoder[new_token_id as usize], new_token_id);

            // TODO: Could probably be improved.
            //
            // Replace occurrences of the most frequent pair in the training
            // corpus with the newly assigned token ID, enabling the discovery
            // and merging of new pairs involving merged tokens.
            let mut i = 0;
            while i < training_corpus.len() - 1 {
                let tokens = &mut training_corpus[i..i + 2];

                if (tokens[0], tokens[1]) == (tok_1, tok_2) {
                    tokens[0] = new_token_id;
                    training_corpus.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Serializing merge rules to disk ensures the vocabulary is
        // reproducible, distributable, and consistent, while preserving
        // training history.
        serialize_rules(merge_rules)?;

        // Insert special tokens to the encoder and decoder.
        for token in SPECIAL_TOKENS {
            let seq: Vec<_> = token
                .as_bytes()
                .iter()
                .map(|byte| *byte as token_id_t)
                .collect();

            let new_token_id = decoder.len() as token_id_t;

            // Can be inserted as is.
            encoder.insert(&seq, new_token_id);
            decoder.push(seq);
        }

        Ok(Self {
            encoder: encoder.flatten(),
            decoder,
        })
    }

    /// Creates a new `Tokenizer` from an existing merge rules file.
    pub fn load(file_path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).open(file_path)?;

        let mut reader = BufReader::new(file);
        let mut line = String::new();

        let mut encoder: Trie = Default::default();
        let mut decoder: Vec<Vec<token_id_t>> = Default::default();

        // Initialize the encoder and decoder with all single-byte tokens
        // (0x00..=0xFF).
        for i in 0x00..=0xFF {
            let token = i as token_id_t;

            // Can be inserted as is.
            encoder.insert(&[token], token);
            decoder.push(vec![token]);
        }

        loop {
            // EOF reached.
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            // Split into exactly two parts: `tok_1` and `tok_2`.
            let mut parts = line.split_whitespace().take(2);

            if let (Some(tok_1), Some(tok_2)) = (parts.next(), parts.next()) {
                let tok_1 = tok_1.parse::<token_id_t>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid merge rules file data")
                })?;

                let tok_2 = tok_2.parse::<token_id_t>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid merge rules file data")
                })?;

                let new_token_id = decoder.len() as token_id_t;

                // Since tokens can themselves be nested merges, the fully
                // expanded token sequences of the token pair should be
                // inserted into the encoder, so it can greedily match the
                // longest possible sequences when encoding new text.
                let mut merged_seq = Vec::with_capacity(
                    decoder[tok_1 as usize].len() + decoder[tok_2 as usize].len(),
                );

                merged_seq.extend_from_slice(&decoder[tok_1 as usize]);
                merged_seq.extend_from_slice(&decoder[tok_2 as usize]);

                decoder.push(merged_seq);
                encoder.insert(&decoder[new_token_id as usize], new_token_id);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid merge rules file data",
                ));
            };

            // Ensure line is cleared each iteration.
            line.clear();
        }

        // Append special tokens to the encoder and decoder.
        for token in SPECIAL_TOKENS {
            let seq: Vec<_> = token
                .as_bytes()
                .iter()
                .map(|byte| *byte as token_id_t)
                .collect();

            let new_token_id = decoder.len() as token_id_t;

            // Can be inserted as is.
            encoder.insert(&seq, new_token_id);
            decoder.push(seq);
        }

        Ok(Self {
            encoder: encoder.flatten(),
            decoder,
        })
    }

    /// Encodes the provided text segment into a token ID sequence.
    pub fn encode(&self, input: &str) -> Vec<token_id_t> {
        let mut encoded = Vec::new();

        // New text segments are encoded as flat token vectors, preserving
        // whitespace for better compression by capturing more merges than if
        // split by spaces.
        let input_seq: Vec<_> = input
            .as_bytes()
            .iter()
            .map(|&byte| byte as token_id_t)
            .collect();

        let mut pos = 0;

        while pos < input_seq.len() {
            let (match_len, maybe_token_id) = self.encoder.longest_match(&input_seq[pos..]);

            if let Some(token_id) = maybe_token_id {
                encoded.push(token_id);
                pos += match_len;
            } else {
                // No matching token sequence found, so insert the token as is.
                encoded.push(input_seq[pos]);
                pos += 1;
            }
        }

        encoded
    }

    /// Decodes the token ID sequence into its original string representation.
    ///
    /// # Panics
    ///
    /// Panics if the decoded bytes are not valid UTF-8.
    pub fn decode(&self, input: &[token_id_t]) -> String {
        let mut output = Vec::new();

        for token_id in input {
            decode_token(*token_id, &self.decoder, &mut output);
        }

        String::from_utf8(output).expect("decoded bytes are not valid UTF-8")
    }
}

/// Serializes the merge rules to disk, allowing the BPE training process to be
/// skipped when creating a [Tokenizer].
///
/// The merges are stored in the following format:
///
/// ```text
/// 100 130
/// 324 443
/// 1080 327
/// ```
///
/// Each line represents a token pair to merge next.
fn serialize_rules(mut merge_rules: VecDeque<(token_id_t, token_id_t)>) -> io::Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open("merges.txt")?;

    let mut writer = BufWriter::new(file);
    let mut line = String::new();

    while let Some((tok_1, tok_2)) = merge_rules.pop_front() {
        write!(&mut line, "{tok_1} {tok_2}")
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        writeln!(&mut writer, "{}", &line[..])?;

        // Ensure line is cleared each iteration.
        line.clear();
    }

    Ok(())
}

/// Recursively decodes token IDs into their byte equivalents.
fn decode_token(token_id: token_id_t, decoder: &[Vec<token_id_t>], output: &mut Vec<u8>) {
    let token_seq = &decoder[token_id as usize];

    for &sub_token in token_seq {
        if sub_token <= 0xFF {
            output.push(sub_token as u8);
        } else {
            // Recursively decode sub-tokens back to their byte representation.
            decode_token(sub_token, decoder, output);
        }
    }
}

/// Counts the frequency of adjacent token pairs.
///
/// # Notes
///
/// Clears the provided `HashMap` before counting.
fn count_pair_freqs(input: &[token_id_t], freq_map: &mut HashMap<u32, u32>) {
    // Reset the frequency map before counting.
    freq_map.clear();

    for tokens in input.windows(2) {
        // NOTE: This assumes a token ID size of u16.
        //
        // The first token occupies the higher 16 bits, the second token
        // occupies the lower 16 bits.
        let key = (tokens[0] as u32) << 16 | tokens[1] as u32;
        *freq_map.entry(key).or_insert(0) += 1;
    }
}
