use std::mem;

use crate::token_id_t;

/// `Trie` for storing token IDs associated with token sequences.
#[derive(Debug, Clone)]
pub struct Trie {
    /// `Trie` nodes.
    nodes: Vec<Node>,
    /// Temporary per-node edge lists to maintain child-node relationships
    /// during insertions.
    tmp_edges: Vec<Vec<Edge>>,
}

/// Flattened `Trie` optimized for efficient lookup of token IDs associated
/// with token sequences.
///
/// Constructed using [Trie::flatten] after all insertions are complete.
#[derive(Debug, Clone)]
pub struct FlattenTrie {
    /// `Trie` nodes.
    nodes: Vec<Node>,
    /// Contiguous edge list representing child-node relationships.
    edges: Vec<Edge>,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    /// Index to the start of the node's edges in `FlattenTrie`.
    child_idx: usize,
    /// Total number of child edges.
    child_len: usize,
    /// Token ID marking the end of a complete token sequence.
    token_id: Option<token_id_t>,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    /// Token label associated with this edge.
    token: token_id_t,
    /// Index of the child node the edge "points" to.
    child_idx: usize,
}

impl Trie {
    /// Creates a new `Trie` with a single root node.
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                child_idx: 0,
                child_len: 0,
                token_id: None,
            }],
            tmp_edges: Default::default(),
        }
    }

    /// Inserts a token sequence into the `Trie`, assigning the terminal node
    /// the provided `token_id`.
    pub fn insert(&mut self, seq: &[token_id_t], token_id: token_id_t) {
        if !seq.is_empty() {
            let mut current_node_idx = 0;

            for token in seq {
                // Check for edges that contain a matching token.
                if let Some(edge_idx) = self
                    .tmp_edges
                    .get(current_node_idx)
                    .and_then(|children| children.iter().position(|edge| edge.token == *token))
                {
                    current_node_idx = self.tmp_edges[current_node_idx][edge_idx].child_idx;
                } else {
                    let node_idx = self.nodes.len();
                    self.nodes.push(Node {
                        child_idx: 0,
                        child_len: 0,
                        token_id: None,
                    });

                    // Ensure each new node gets a unique edge vector.
                    //
                    // Pushing instead of inserting preserves the position of
                    // existing edge vectors.
                    self.tmp_edges.push(Vec::new());
                    self.tmp_edges[current_node_idx].push(Edge {
                        token: *token,
                        child_idx: node_idx,
                    });

                    self.nodes[current_node_idx].child_len += 1;

                    current_node_idx = node_idx;
                }
            }

            self.nodes[current_node_idx].token_id = Some(token_id);
        }
    }

    /// Consumes this `Trie`, flattening its temporary edge lists into a
    /// contiguous edge array.
    ///
    /// Returns a `FlattenTrie`, optimized for token ID lookups.
    ///
    /// # Notes
    ///
    /// Must be called after all insertions are complete.
    pub fn flatten(mut self) -> FlattenTrie {
        let mut flat_trie = FlattenTrie {
            nodes: Default::default(),
            edges: Default::default(),
        };

        let mut offset = 0;

        for (i, mut node) in self.nodes.into_iter().enumerate() {
            if let Some(edges) = self.tmp_edges.get_mut(i) {
                flat_trie.edges.append(&mut mem::take(edges));
            }

            node.child_idx = offset;
            offset += node.child_len;

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
    /// Finds the length of the longest prefix of the token sequence present in
    /// the `Trie`, along with the token ID associated with the terminal node,
    /// if any exists.
    pub fn longest_match(&self, seq: &[token_id_t]) -> (usize, Option<token_id_t>) {
        let mut matched_count = 0;
        let mut token_id = None;

        let mut current_node_idx = 0;

        for token in seq {
            let node = &self.nodes[current_node_idx];
            let first_child = node.child_idx;
            let child_count = node.child_len;

            // Check for edges that contain a matching token.
            if let Some(edge) = &self.edges[first_child..first_child + child_count]
                .iter()
                .find(|entry| entry.token == *token)
            {
                current_node_idx = edge.child_idx;

                // Store token ID of the current node (not the previous one).
                token_id = self.nodes[current_node_idx].token_id;
                matched_count += 1;
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
    fn trie_empty_longest_match() {
        let trie = Trie::new().flatten();

        let (len, token) = trie.longest_match(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_single_insert() {
        let mut trie = Trie::new();
        trie.insert(&[0xAB, 0xCD, 0xEF], 100);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0xAB, 0xCD, 0xEF]);
        assert_eq!(len, 3);
        assert_eq!(token, Some(100));
    }

    #[test]
    fn trie_single_insert_partial_match() {
        let mut trie = Trie::new();
        trie.insert(&[0xAB, 0xCD, 0xEF], 100);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0xAB, 0xCD]);
        assert_eq!(len, 2);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_multiple_insert_longest_match() {
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
    fn trie_no_match() {
        let mut trie = Trie::new();
        trie.insert(&[0xAB, 0xCD], 10);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0xDE, 0xAD]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_multiple_insert_disjoint() {
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
    fn trie_common_prefix_insert() {
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
    fn trie_empty_sequence() {
        let mut trie = Trie::new();
        trie.insert(&[], 999);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[]);
        assert_eq!(len, 0);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_overlapping_insert() {
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
    fn trie_partial_match_no_token() {
        let mut trie = Trie::new();
        trie.insert(&[0x10, 0x20, 0x30, 0x40], 100);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0x10, 0x20]);
        assert_eq!(len, 2);
        assert_eq!(token, None);
    }

    #[test]
    fn trie_partial_match_token() {
        let mut trie = Trie::new();
        trie.insert(&[0x10, 0x20], 101);

        let trie = trie.flatten();

        let (len, token) = trie.longest_match(&[0x10, 0x20, 0x30]);
        assert_eq!(len, 2);
        assert_eq!(token, Some(101));
    }

    // NOTE: This assumes a token ID size of u16.
    #[test]
    fn trie_nested_sequence_insert() {
        let mut trie = Trie::new();

        for length in 1u16..=20 {
            let seq: Vec<_> = (0u16..length).map(|i| i * 3 + 7).collect();
            trie.insert(&seq, length);
        }

        let trie = trie.flatten();

        for length in 1u16..=20 {
            let seq: Vec<_> = (0u16..length).map(|i| i * 3 + 7).collect();
            let (len, token) = trie.longest_match(&seq);
            assert_eq!(len, length as usize);
            assert_eq!(token, Some(length));
        }

        let seq: Vec<_> = (0u16..19).map(|i| i * 3 + 7).collect();
        let (len, token) = trie.longest_match(&seq);
        assert_eq!(len, 19);
        assert_eq!(token, Some(19));

        let mut seq: Vec<_> = (0u16..21).map(|i| i * 3 + 7).collect();
        seq[20] = 0xFF;
        let (len, token) = trie.longest_match(&seq);
        assert_eq!(len, 20);
        assert_eq!(token, Some(20));
    }
}
