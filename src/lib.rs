//! [Byte-Pair Encoding] (BPE) implementation from scratch, designed
//! specifically for tokenization tasks in NLP and text processing.
//!
//! [Byte-Pair Encoding]: https://en.wikipedia.org/wiki/Byte-pair_encoding

#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]

mod trie;

use std::collections::HashMap;
use trie::{FlattenTrie, Trie};

/// The size of a token ID for the `Tokenizer`.
///
/// Allows for up to ~65K unique token sequences.
#[allow(non_camel_case_types)]
pub type token_id_t = u16;

/// The maximum `vocabulary` size, considering token ID size and initial
/// single-byte tokens.
const MAX_VOCAB_SIZE: token_id_t = token_id_t::MAX - 256;

/// A byte-level [BPE] tokenizer, used to build a `vocabulary` of subword units
/// from a raw corpus.
///
/// Instead of fixed word/character tokens, BPE learns frequently co-occurring
/// sequences of bytes/characters from the data. This enables models to handle
/// rare, unseen, or unknown characters by representing text as combinations of
/// subwords.
///
/// [BPE]: https://en.wikipedia.org/wiki/Byte-pair_encoding
#[derive(Debug)]
pub struct Tokenizer {
    /// Stores the `vocabulary` in a `FlattenTrie` for efficient text encoding.
    encoder: FlattenTrie,
    /// Maps token IDs to their corresponding token sequences.
    decoder: Vec<Vec<token_id_t>>,
}

impl Tokenizer {
    /// Creates a new `Tokenizer` from the provided corpus.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided `raw_corpus` is empty.
    pub fn new(raw_corpus: &str) -> Self {
        // `Trie` used for efficient encoding by greedily matching the longest
        // prefix of the given input.
        let mut encoder: Trie = Default::default();

        // Vector of token sequences used for decoding, allowing quick indexing
        // of token sequences via token ID.
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
            encoder.insert(&[token], token);
            decoder.push(vec![token]);
        }

        // Split the corpus on whitespace, convert each word to byte slices,
        // and then map each byte slice to an owned token sequence.
        //
        // Splitting by whitespace ensures byte frequencies are learned within
        // words, not across word boundaries. Also, since BPE merges frequent
        // byte sequences into subwords, each element must be able to represent
        // both base bytes and token IDs.
        let mut training_corpus: Vec<_> = raw_corpus
            .split_whitespace()
            .map(|word| {
                word.as_bytes()
                    .iter()
                    .map(|byte| *byte as token_id_t)
                    .collect()
            })
            .collect();

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
                .expect("provided empty intial corpus");

            // Stop if no more tokens can be merged or the vocabulary is full.
            if *count < 2 || decoder.len() >= MAX_VOCAB_SIZE as usize {
                break;
            }

            // Unpack the token pair.
            //
            // NOTE: This assumes a token ID size of u16.
            let first_tok = (pair >> 16) as token_id_t;
            let second_tok = (pair & 0xFFFF) as token_id_t;

            // The new ID is based on the decoder length, incrementing from 256
            // after the initial single-byte tokens are inserted.
            let new_token_id = decoder.len() as token_id_t;

            encoder.insert(&[first_tok, second_tok], new_token_id);
            decoder.push(vec![first_tok, second_tok]);

            // TODO: Could probably be improved.
            //
            // Replace occurrences of the most frequent pair in the training
            // corpus with the newly assigned token ID, enabling the discovery
            // and merging of new pairs involving the merged token.
            for word in training_corpus.iter_mut() {
                let mut i = 0;

                while i + 1 < word.len() {
                    if (word[i], word[i + 1]) == (first_tok, second_tok) {
                        word[i] = new_token_id;
                        word.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        Self {
            encoder: encoder.flatten(),
            decoder,
        }
    }

    /// Encodes the provided text segment into its corresponding token ID
    /// sequence.
    pub fn encode(&self, input: &str) -> Vec<token_id_t> {
        // Convert the input to bytes, then map them to a sequence of token
        // IDs.
        let input_seq: Vec<_> = input
            .as_bytes()
            .iter()
            .map(|byte| *byte as token_id_t)
            .collect();

        let mut encoded = Vec::new();
        let mut pos = 0;

        while pos < input_seq.len() {
            let (match_len, maybe_token_id) = self.encoder.longest_match(&input_seq[pos..]);

            if let Some(token_id) = maybe_token_id {
                encoded.push(token_id);
                pos += match_len;
            } else {
                // No matching token ID found, so insert the token as is.
                encoded.push(input_seq[pos]);
                pos += 1;
            }
        }

        encoded
    }

    /// Decodes the provided sequence of token IDs into its original string
    /// representation.
    ///
    /// # Panics
    ///
    /// This function will panic if the decoded bytes are not valid UTF-8.
    pub fn decode(&self, input: &[token_id_t]) -> String {
        let mut output = Vec::new();

        for &token_id in input {
            decode_token(token_id, &self.decoder, &mut output);
        }

        String::from_utf8(output).expect("decoded bytes are not valid UTF-8")
    }
}

// Recursively decodes token IDs into their byte equivalents.
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
/// Clears the provided [HashMap] before counting.
fn count_pair_freqs(input: &[Vec<token_id_t>], freq_map: &mut HashMap<u32, u32>) {
    // Reset the frequency map before counting.
    freq_map.clear();

    for word in input {
        // Compare adjacent tokens.
        for tokens in word.windows(2) {
            // NOTE: This assumes a token ID size of u16.
            //
            // The first token occupies the higher 16 bits, the second token
            // occupies the lower 16 bits.
            let key = (tokens[0] as u32) << 16 | tokens[1] as u32;
            *freq_map.entry(key).or_insert(0) += 1;
        }
    }
}
