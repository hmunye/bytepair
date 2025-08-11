use std::mem;

/// `Trie` for storage of token IDs associated with byte sequences.
#[derive(Debug, Clone)]
pub struct Trie {
    /// Nodes in the trie.
    nodes: Vec<Node>,
    /// Temporary per-node edge lists used during insertions.
    tmp_edges: Vec<Vec<Edge>>,
}

/// Flattened `Trie` optimized for efficient lookup of token IDs associated
/// with byte sequences.
///
/// Constructed using [Trie::flatten] after all insertions are complete.
#[derive(Debug, Clone)]
pub struct FlattenTrie {
    /// Nodes in the flattened trie.
    nodes: Vec<Node>,
    /// Contiguous edge list representing child links between nodes.
    edges: Vec<Edge>,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    /// Index of the first child edge in the flattened `edges`.
    child_idx: u32,
    /// Number of child edges.
    children_len: u32,
    /// Token ID marking the end of a complete byte sequence.
    token_id: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    /// Byte label associated with this edge (0–255).
    byte: u8,
    /// Index of the child node this edge points to.
    child_idx: u32,
}

impl Trie {
    /// Creates a new `Trie` with a single root node.
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                child_idx: 0,
                children_len: 0,
                token_id: None,
            }],
            tmp_edges: Default::default(),
        }
    }

    /// Inserts a byte sequence into the `Trie`, assigning the terminal node
    /// the provided `token_id`.
    pub fn insert(&mut self, byte_seq: &[u8], token_id: u32) {
        if !byte_seq.is_empty() {
            let mut current_node_idx = 0;

            for byte in byte_seq {
                // Check for existing edge matching the byte.
                if let Some(edge_idx) = self
                    .tmp_edges
                    .get(current_node_idx)
                    .and_then(|children| children.iter().position(|edge| edge.byte == *byte))
                {
                    current_node_idx =
                        self.tmp_edges[current_node_idx][edge_idx].child_idx as usize;
                } else {
                    // Create a new node and edge for the unmatched byte.
                    let node_idx = self.nodes.len();
                    self.nodes.push(Node {
                        child_idx: 0,
                        children_len: 0,
                        token_id: None,
                    });

                    // Ensure new node has it's own edge vector.
                    self.tmp_edges.push(Vec::new());
                    self.tmp_edges[current_node_idx].push(Edge {
                        byte: *byte,
                        child_idx: node_idx as u32,
                    });

                    self.nodes[current_node_idx].children_len += 1;
                    current_node_idx = node_idx;
                }
            }

            self.nodes[current_node_idx].token_id = Some(token_id);
        }
    }

    /// Consumes this `Trie`, flattening its temporary edge lists into
    /// a contiguous edge array.
    ///
    /// Returns a [`FlattenTrie`], optimized for token ID lookups.
    ///
    /// # Notes
    ///
    /// Must be called after all insertions are complete.
    pub fn flatten(mut self) -> FlattenTrie {
        let mut flat_trie = FlattenTrie {
            nodes: Default::default(),
            edges: Default::default(),
        };

        // Track position within `edges`, for node child index updating.
        let mut offset = 0;

        for (i, mut node) in self.nodes.into_iter().enumerate() {
            if let Some(edges) = self.tmp_edges.get_mut(i) {
                flat_trie.edges.append(&mut mem::take(edges));
            }

            node.child_idx = offset;
            offset += node.children_len;

            flat_trie.nodes.push(node);
        }

        flat_trie
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl FlattenTrie {
    /// Finds the length of the longest prefix of the byte sequence present in
    /// the `Trie`, along with the token ID associated with the terminal node,
    /// if any exists.
    pub fn longest_match(&self, byte_seq: &[u8]) -> (usize, Option<u32>) {
        let mut matched_count = 0;
        let mut token_id = None;

        let mut current_node_idx = 0;

        for byte in byte_seq {
            let node = &self.nodes[current_node_idx];

            let first_child = node.child_idx as usize;
            let child_count = node.children_len as usize;

            if let Some(edge) = &self.edges[first_child..first_child + child_count]
                .iter()
                .find(|entry| entry.byte == *byte)
            {
                current_node_idx = edge.child_idx as usize;

                matched_count += 1;
                token_id = self.nodes[current_node_idx].token_id;
            } else {
                break;
            }
        }

        (matched_count, token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_empty_longest_match_returns_none() {
        let trie = Trie::new().flatten();

        let (len, token) = trie.longest_match(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_single_insertion_exact_match() {
        let mut trie = Trie::new();
        trie.insert(&[0xAB, 0xCD, 0xEF], 100);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0xAB, 0xCD, 0xEF]);
        assert_eq!(len, 3);
        assert_eq!(token, Some(100));
    }

    #[test]
    fn trie_single_insertion_partial_match() {
        let mut trie = Trie::new();
        trie.insert(&[0xAB, 0xCD, 0xEF], 100);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0xAB, 0xCD]);
        assert_eq!(len, 2);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_multiple_insertions_longest_match_returns_correct_token() {
        let mut trie = Trie::new();
        trie.insert(&[0x10, 0x20], 200);
        trie.insert(&[0x10, 0x20, 0x30], 201);
        trie.insert(&[0x10, 0x30], 202);

        let trie = trie.flatten();

        let cases = [
            (&[0x10, 0x20, 0x30, 0x40][..], 3, Some(201)),
            (&[0x10, 0x20][..], 2, Some(200)),
            (&[0x10, 0x30][..], 2, Some(202)),
            (&[0x10][..], 1, None),
        ];

        for &(input, expected_len, expected_token) in &cases {
            let (len, token) = trie.longest_match(input);
            assert_eq!(len, expected_len);
            assert_eq!(token, expected_token);
        }
    }

    #[test]
    fn trie_no_match_returns_none() {
        let mut trie = Trie::new();
        trie.insert(&[0xAB, 0xCD], 10);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0xDE, 0xAD]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_insert_multiple_disjoint_sequences() {
        let mut trie = Trie::new();
        trie.insert(&[0xAA, 0xBB], 1);
        trie.insert(&[0xCC, 0xDD], 2);
        trie.insert(&[0xEE], 3);

        let trie = trie.flatten();

        let cases = [
            (&[0xAA, 0xBB][..], 2, Some(1)),
            (&[0xCC, 0xDD][..], 2, Some(2)),
            (&[0xEE][..], 1, Some(3)),
            (&[0xFF][..], 0, None),
        ];

        for &(input, expected_len, expected_token) in &cases {
            let (len, token) = trie.longest_match(input);
            assert_eq!(len, expected_len);
            assert_eq!(token, expected_token);
        }
    }

    #[test]
    fn trie_insert_sequences_with_common_prefixes() {
        let mut trie = Trie::new();
        trie.insert(&[0x11], 10);
        trie.insert(&[0x11, 0x22], 20);
        trie.insert(&[0x11, 0x22, 0x33], 30);
        trie.insert(&[0x11, 0x44], 40);

        let trie = trie.flatten();

        let cases = [
            (&[0x11][..], 1, Some(10)),
            (&[0x11, 0x22][..], 2, Some(20)),
            (&[0x11, 0x22, 0x33][..], 3, Some(30)),
            (&[0x11, 0x44][..], 2, Some(40)),
            (&[0x11, 0x22, 0x33, 0xFF][..], 3, Some(30)),
        ];

        for &(input, expected_len, expected_token) in &cases {
            let (len, token) = trie.longest_match(input);
            assert_eq!(len, expected_len);
            assert_eq!(token, expected_token);
        }
    }

    #[test]
    fn trie_empty_sequence_insertion_and_lookup() {
        let mut trie = Trie::new();
        trie.insert(&[], 999);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_insertion_of_overlapping_sequences_with_different_tokens() {
        let mut trie = Trie::new();
        trie.insert(&[0x55, 0x66, 0x77], 43);
        trie.insert(&[0x55, 0x77], 44);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0x55, 0x77]);
        assert_eq!(len, 2);
        assert_eq!(token, Some(44));

        let (len, token) = trie.longest_match(&[0x55, 0x66, 0x77]);
        assert_eq!(len, 3);
        assert_eq!(token, Some(43));
    }

    #[test]
    fn trie_longest_match_with_no_terminal_node_in_prefix() {
        let mut trie = Trie::new();
        trie.insert(&[0x10, 0x20, 0x30, 0x40], 100);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0x10, 0x20]);
        assert_eq!(len, 2);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_longest_match_with_partial_non_terminal_prefix() {
        let mut trie = Trie::new();
        trie.insert(&[0x10, 0x20], 101);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0x10, 0x20, 0x30]);
        assert_eq!(len, 2);
        assert_eq!(token, Some(101));
    }

    #[test]
    fn trie_multiple_random_insertions_and_lookups() {
        let mut trie = Trie::new();

        let sequences: Vec<(&[u8], u32)> = vec![
            (&[0xFF, 0x00, 0xAB], 1),
            (&[0xAB, 0xCD], 2),
            (&[0xDE, 0xAD, 0xBE, 0xEF], 3),
            (&[0x01], 4),
            (&[0x01, 0x02, 0x03, 0x04], 5),
            (&[0x10, 0x20, 0x30], 6),
            (&[0x10, 0x21, 0x30], 7),
        ];

        for &(seq, token) in &sequences {
            trie.insert(seq, token);
        }

        let trie = trie.flatten();

        for &(seq, token) in &sequences {
            let (len, found_token) = trie.longest_match(seq);
            assert_eq!(len, seq.len());
            assert_eq!(found_token, Some(token));
        }

        let (len, token) = trie.longest_match(&[0xFF, 0x00]);
        assert_eq!(len, 2);
        assert_eq!(token, None);

        let (len, token) = trie.longest_match(&[0x00, 0xFF]);
        assert_eq!(len, 0);
        assert_eq!(token, None);

        let (len, token) = trie.longest_match(&[0x10, 0x21, 0x30]);
        assert_eq!(len, 3);
        assert_eq!(token, Some(7));

        let (len, token) = trie.longest_match(&[0x10, 0x20, 0x30]);
        assert_eq!(len, 3);
        assert_eq!(token, Some(6));

        let (len, token) = trie.longest_match(&[0x01, 0x02, 0x03, 0x04, 0x05]);
        assert_eq!(len, 4);
        assert_eq!(token, Some(5));
    }

    #[test]
    fn trie_deeply_nested_sequences() {
        let mut trie = Trie::new();

        for length in 1..=20 {
            let seq: Vec<u8> = (0..length).map(|i| (i * 3 + 7) as u8).collect();
            trie.insert(&seq, length as u32);
        }

        let trie = trie.flatten();

        for length in 1..=20 {
            let seq: Vec<u8> = (0..length).map(|i| (i * 3 + 7) as u8).collect();
            let (len, token) = trie.longest_match(&seq);
            assert_eq!(len, length);
            assert_eq!(token, Some(length as u32));
        }

        let seq: Vec<u8> = (0..19).map(|i| (i * 3 + 7) as u8).collect();
        let (len, token) = trie.longest_match(&seq);
        assert_eq!(len, 19);
        assert_eq!(token, Some(19));

        let mut seq: Vec<u8> = (0..21).map(|i| (i * 3 + 7) as u8).collect();
        seq[20] = 0xFF;
        let (len, token) = trie.longest_match(&seq);
        assert_eq!(len, 20);
        assert_eq!(token, Some(20));
    }

    #[test]
    fn trie_insert_and_match_with_full_byte_range() {
        let mut trie = Trie::new();

        for b in 0..=255u8 {
            let seq = [b];
            trie.insert(&seq, b as u32);
        }

        let trie = trie.flatten();

        for b in 0..=255u8 {
            let (len, token) = trie.longest_match(&[b]);
            assert_eq!(len, 1);
            assert_eq!(token, Some(b as u32));
        }

        let (len, token) = trie.longest_match(&[]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }
}
