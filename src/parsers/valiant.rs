// Valiant parser - equivalent to python_files/final_parser/Valiant.py

use crate::grammars::{Grammar, NumProduction, NumSymbol};
use crate::parse_tree::{ParseSymbol, ParseTree};
use std::collections::{HashMap, HashSet};

pub type ForestNode = Vec<(u32, Vec<ForestRef>)>;

/// Reference to a forest node in the table, or a terminal leaf
#[derive(Clone, Debug)]
pub enum ForestRef {
    /// Reference to table[row][col][nt]
    TableRef { row: usize, col: usize, nt: u32 },
    /// Terminal leaf node
    Terminal(u32),
}

/// Valiant Parse Table: table[s][e] maps non-terminal -> list of (nt, children) derivations
pub type ValiantTable = Vec<Vec<HashMap<u32, ForestNode>>>;

/// A matrix of non-terminal sets (used in Valiant's algorithm)
/// matrix[i][j] = set of non-terminals that can derive input[i..j]
pub type NTMatrix = Vec<Vec<HashSet<u32>>>;

/// A boolean matrix for a single non-terminal
pub type BoolMatrix = Vec<Vec<bool>>;

/// Maps non-terminal ID -> boolean matrix
pub type BoolMatrices = HashMap<u32, BoolMatrix>;

/// Result of multiplying boolean matrix pairs: r[nt1][nt2] = bool matrix
pub type PairProduct = HashMap<u32, HashMap<u32, BoolMatrix>>;

pub struct ValiantParser {
    pub cache: HashMap<(u64, usize), NTMatrix>,
    pub cell_width: usize,
    pub grammar: Grammar,
    pub productions: Vec<(u32, NumProduction)>,
    pub terminal_productions: Vec<(u32, u32)>,
    pub nonterminal_productions: Vec<(u32, (u32, u32))>,
    pub nonterminals: Vec<u32>,
}

impl ValiantParser {
    pub fn new(grammar: Grammar) -> Self {
        let mut productions: Vec<(u32, NumProduction)> = Vec::new();
        let mut terminal_productions: Vec<(u32, u32)> = Vec::new();
        let mut nonterminal_productions: Vec<(u32, (u32, u32))> = Vec::new();

        // Collect all non-terminals
        let mut nonterminals: Vec<u32> = grammar.rules.keys().copied().collect();
        nonterminals.sort();

        // Collect all productions
        for (&lhs, rhs_list) in &grammar.rules {
            for rhs in rhs_list {
                productions.push((lhs, rhs.clone()));

                // Classify by type
                if rhs.len() == 1 {
                    if let NumSymbol::Terminal(t) = rhs[0] {
                        terminal_productions.push((lhs, t));
                    }
                } else if rhs.len() == 2 {
                    if let (NumSymbol::NonTerminal(nt1), NumSymbol::NonTerminal(nt2)) =
                        (&rhs[0], &rhs[1])
                    {
                        nonterminal_productions.push((lhs, (*nt1, *nt2)));
                    }
                }
            }
        }

        ValiantParser {
            cache: HashMap::new(),
            cell_width: 5,
            grammar,
            productions,
            terminal_productions,
            nonterminal_productions,
            nonterminals,
        }
    }

    fn init_table(&self, length: usize) -> ValiantTable {
        // We need table[s][e] for 0 <= s <= e <= length
        // Using length+1 for both dimensions
        vec![vec![HashMap::new(); length + 1]; length + 1]
    }

    fn parse_1(&self, text: &[u32], length: usize, table: &mut ValiantTable) {
        for s in 0..length {
            for &(key, terminal) in &self.terminal_productions {
                if text[s] == terminal {
                    let entry = table[s][s + 1].entry(key).or_insert_with(Vec::new);
                    entry.push((key, vec![ForestRef::Terminal(text[s])]));
                }
            }
        }
    }

    /// Compute a hash for a matrix (used as cache key)
    /// Python uses str(A) but we need something hashable in Rust
    fn matrix_hash(&self, a: &NTMatrix) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for row in a {
            for cell in row {
                // Sort the elements for consistent hashing
                let mut elements: Vec<_> = cell.iter().collect();
                elements.sort();
                for &nt in elements {
                    nt.hash(&mut hasher);
                }
                // Separator between cells
                u32::MAX.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Multiply two subsets of non-terminals using grammar productions
    /// Python: def multiply_subsets(N1, N2, P):
    ///     return {Ai:True for Ai, (Aj,Ak) in P if Aj in N1 and Ak in N2}
    // fn multiply_subsets(&self, n1: &HashSet<u32>, n2: &HashSet<u32>) -> HashSet<u32> {
    //     let mut result = HashSet::new();
    //     for &(ai, (aj, ak)) in &self.nonterminal_productions {
    //         if n1.contains(&aj) && n2.contains(&ak) {
    //             result.insert(ai);
    //         }
    //     }
    //     result
    // }

    // /// Multiply two matrices of non-terminal sets
    // /// Python: def multiply_matrices(A, B, P):
    // fn multiply_matrices(&self, a: &NTMatrix, b: &NTMatrix) -> NTMatrix {
    //     let m = a.len();
    //     let mut c: NTMatrix = vec![vec![HashSet::new(); m]; m];

    //     for i in 0..m {
    //         for j in 0..m {
    //             for k in 0..m {
    //                 let product = self.multiply_subsets(&a[i][k], &b[k][j]);
    //                 c[i][j].extend(product);
    //             }
    //         }
    //     }
    //     c
    // }

    /// Convert a matrix of non-terminal sets to boolean matrices (one per non-terminal)
    /// Python: def bool_matrices(A, nonterminals):
    fn bool_matrices(&self, a: &NTMatrix) -> BoolMatrices {
        let m = a.len();
        let mut m_ks: BoolMatrices = HashMap::new();

        // Initialize boolean matrices for each non-terminal
        for &nt in &self.nonterminals {
            m_ks.insert(nt, vec![vec![false; m]; m]);
        }

        // Fill in true values where non-terminals appear
        for i in 0..m {
            for j in 0..m {
                for &nt in &a[i][j] {
                    if let Some(matrix) = m_ks.get_mut(&nt) {
                        matrix[i][j] = true;
                    }
                }
            }
        }

        m_ks
    }

    /// Multiply two boolean matrices
    /// Python: def multiply_bool_matrices(A, B):
    fn multiply_bool_matrices(&self, a: &BoolMatrix, b: &BoolMatrix) -> BoolMatrix {
        let m = a.len();
        let mut c: BoolMatrix = vec![vec![false; m]; m];

        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    if a[i][k] && b[k][j] {
                        c[i][j] = true;
                        break;
                    }
                }
            }
        }
        c
    }

    /// Multiply all pairs of boolean matrices
    /// Python: def multiply_pairs(bool_As, bool_Bs):
    fn multiply_pairs(&self, bool_as: &BoolMatrices, bool_bs: &BoolMatrices) -> PairProduct {
        let mut r: PairProduct = HashMap::new();

        for (&a_key, a_matrix) in bool_as {
            let mut inner: HashMap<u32, BoolMatrix> = HashMap::new();
            for (&b_key, b_matrix) in bool_bs {
                inner.insert(b_key, self.multiply_bool_matrices(a_matrix, b_matrix));
            }
            r.insert(a_key, inner);
        }
        r
    }

    fn get_final_matrix(&self, r: &PairProduct, m: usize) -> NTMatrix {
        let mut result: NTMatrix = vec![vec![HashSet::new(); m]; m];

        for i in 0..m {
            for j in 0..m {
                for &(ai, (aj, ak)) in &self.nonterminal_productions {
                    if let Some(inner) = r.get(&aj) {
                        if let Some(matrix) = inner.get(&ak) {
                            if matrix[i][j] {
                                result[i][j].insert(ai);
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Multiply matrices using boolean matrix decomposition (Valiant's optimization)
    /// Python: def multiply_matrices_b(A, B, P, nonterminals):
    fn multiply_matrices_b(&self, a: &NTMatrix, b: &NTMatrix) -> NTMatrix {
        let length = a.len();
        let bool_as = self.bool_matrices(a);
        let bool_bs = self.bool_matrices(b);
        let r = self.multiply_pairs(&bool_as, &bool_bs);
        self.get_final_matrix(&r, length)
    }

    /// Union of two matrices
    /// Python: def union_matrices(A, B):
    fn union_matrices(&self, a: &NTMatrix, b: &NTMatrix) -> NTMatrix {
        let m = a.len();
        let mut c: NTMatrix = vec![vec![HashSet::new(); m]; m];

        for i in 0..m {
            for j in 0..m {
                c[i][j].extend(&a[i][j]);
                c[i][j].extend(&b[i][j]);
            }
        }
        c
    }

    fn parsed_in_steps(&mut self, a: &NTMatrix, i: usize) -> NTMatrix {
        // Base case: i == 1
        if i == 1 {
            return a.clone();
        }

        // Check cache
        let hash = self.matrix_hash(a);
        if let Some(cached) = self.cache.get(&(hash, i)) {
            return cached.clone();
        }

        // Initialize result matrix
        let m = a.len();
        let mut res: NTMatrix = vec![vec![HashSet::new(); m]; m];

        // For j in range(1, i)
        for j in 1..i {
            let a_j = self.parsed_in_steps(a, j);
            let b = self.parsed_in_steps(a, i - j);
            let product = self.multiply_matrices_b(&a_j, &b);
            res = self.union_matrices(&res, &product);
        }

        // Cache the result
        self.cache.insert((hash, i), res.clone());

        res
    }

    fn transitive_closure(&mut self, a: &NTMatrix, l: usize) -> NTMatrix {
        let m = a.len();
        let mut res: NTMatrix = vec![vec![HashSet::new(); m]; m];

        // For i in range(1, l+1)
        for i in 1..=l {
            let a_i = self.parsed_in_steps(a, i);
            res = self.union_matrices(&res, &a_i);
        }

        res
    }

    pub fn recognize_on(&mut self, text: &[u32], start_symbol: u32) -> bool {
        let n = text.len();
        if n == 0 {
            return false;
        }

        // Initialize and fill the base table (terminals)
        let mut table = self.init_nt_matrix(n);
        self.fill_terminals(text, n, &mut table);

        // Compute transitive closure
        let closure = self.transitive_closure(&table, n);

        // Check if start symbol is in closure[0][n]
        closure[0][n].contains(&start_symbol)
    }

    /// Initialize an NTMatrix (for recognition, not parse forest)
    fn init_nt_matrix(&self, length: usize) -> NTMatrix {
        vec![vec![HashSet::new(); length + 1]; length + 1]
    }

    /// Fill terminal productions into NTMatrix
    fn fill_terminals(&self, text: &[u32], length: usize, table: &mut NTMatrix) {
        for s in 0..length {
            for &(key, terminal) in &self.terminal_productions {
                if text[s] == terminal {
                    table[s][s + 1].insert(key);
                }
            }
        }
    }

    // ========================================================================
    // Parse Tree Construction
    // ========================================================================

    /// Extract a parse tree from the parse table
    fn trees(&self, table: &ValiantTable, forest_ref: &ForestRef) -> Option<ParseTree> {
        match forest_ref {
            ForestRef::Terminal(t) => Some(ParseTree::new(
                ParseSymbol::Terminal(self.grammar.terminals.get_str(*t).unwrap().to_string()),
                vec![],
            )),

            ForestRef::TableRef { row, col, nt } => {
                let forest_node = table[*row][*col].get(nt)?;
                if forest_node.is_empty() {
                    return None;
                }

                let (key, children) = &forest_node[0];

                let mut child_trees = Vec::new();
                for child_ref in children {
                    if let Some(child_tree) = self.trees(table, child_ref) {
                        child_trees.push(child_tree);
                    }
                }

                Some(ParseTree {
                    name: ParseSymbol::NonTerminal(
                        self.grammar.non_terminal_str(*key).unwrap().to_string(),
                    ),
                    children: child_trees,
                })
            }
        }
    }

    /// Main parsing function
    pub fn parse_on(&self, text: &[u32], start_symbol: u32) -> Option<ParseTree> {
        let length = text.len();
        if length == 0 {
            return None;
        }

        let mut table = self.init_table(length);
        self.parse_1(text, length, &mut table);

        // TODO: Implement full Valiant parsing with matrix multiplication
        // For now, fall back to CYK-style parsing
        for n in 2..=length {
            self.parse_n(n, length, &mut table);
        }

        if table[0][length].contains_key(&start_symbol) {
            let root_ref = ForestRef::TableRef {
                row: 0,
                col: length,
                nt: start_symbol,
            };
            return self.trees(&table, &root_ref);
        }

        None
    }

    /// CYK-style parsing for spans of length n (fallback)
    fn parse_n(&self, n: usize, length: usize, table: &mut ValiantTable) {
        for s in 0..=length - n {
            for p in 1..n {
                for &(k, (r_b, r_c)) in &self.nonterminal_productions {
                    let has_left = table[s][s + p].contains_key(&r_b);
                    let has_right = table[s + p][s + n].contains_key(&r_c);

                    if has_left && has_right {
                        let left_ref = ForestRef::TableRef {
                            row: s,
                            col: s + p,
                            nt: r_b,
                        };
                        let right_ref = ForestRef::TableRef {
                            row: s + p,
                            col: s + n,
                            nt: r_c,
                        };

                        let entry = table[s][s + n].entry(k).or_insert_with(Vec::new);
                        entry.push((k, vec![left_ref, right_ref]));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::load_grammar_from_str;

    #[test]
    fn test_valiant_simple_grammar() {
        let json = r#"{
            "name": "simple",
            "start": "<S>",
            "rules": {
                "<S>": [["<A>", "<B>"]],
                "<A>": [["a"]],
                "<B>": [["b"]]
            }
        }"#;

        let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
        let cnf = grammar.to_cnf();

        println!("=== Testing Valiant Parser (simple) ===");
        cnf.debug_print();

        let token_a = cnf.terminals.get_id("a").expect("Token 'a' not found");
        let token_b = cnf.terminals.get_id("b").expect("Token 'b' not found");
        let input = vec![token_a, token_b];

        println!("Input tokens: {:?} (a={}, b={})", input, token_a, token_b);

        let parser = ValiantParser::new(cnf);
        let result = parser.parse_on(&input, parser.grammar.start);

        match &result {
            Some(tree) => {
                println!("\n✓ Parse successful!");
                println!("\nParse tree:");
                println!("{}", tree.display());
            }
            None => {
                println!("\n✗ Parse failed!");
            }
        }

        assert!(result.is_some(), "Should parse 'a b' successfully");
    }

    #[test]
    fn test_valiant_longer_input() {
        let json = r#"{
            "name": "longer",
            "start": "<S>",
            "rules": {
                "<S>": [["<A>", "<B>"], ["<S>", "<S>"]],
                "<A>": [["a"]],
                "<B>": [["b"]]
            }
        }"#;

        let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
        let cnf = grammar.to_cnf();

        println!("\n=== Testing Valiant Parser (longer input) ===");
        cnf.debug_print();

        let token_a = cnf.terminals.get_id("a").expect("Token 'a' not found");
        let token_b = cnf.terminals.get_id("b").expect("Token 'b' not found");
        let input = vec![token_a, token_b, token_a, token_b];

        println!("Input tokens: {:?}", input);

        let parser = ValiantParser::new(cnf);
        let result = parser.parse_on(&input, parser.grammar.start);

        match &result {
            Some(tree) => {
                println!("\n✓ Parse successful!");
                println!("\nParse tree:");
                println!("{}", tree.display());
            }
            None => {
                println!("\n✗ Parse failed!");
            }
        }

        assert!(result.is_some(), "Should parse 'a b a b' successfully");
    }

    #[test]
    fn test_valiant_reject_invalid() {
        let json = r#"{
            "name": "simple",
            "start": "<S>",
            "rules": {
                "<S>": [["<A>", "<B>"]],
                "<A>": [["a"]],
                "<B>": [["b"]]
            }
        }"#;

        let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
        let cnf = grammar.to_cnf();

        let token_a = cnf.terminals.get_id("a").expect("Token 'a' not found");
        let input = vec![token_a, token_a]; // "a a" - should fail

        println!("\n=== Testing Valiant Parser (invalid input) ===");
        println!("Input tokens: {:?} (should be rejected)", input);

        let parser = ValiantParser::new(cnf);
        let result = parser.parse_on(&input, parser.grammar.start);

        assert!(result.is_none(), "Should reject 'a a'");
        println!("✓ Correctly rejected invalid input");
    }
}

/// Parse input using Valiant algorithm
/// Returns true if the input (as numeric token IDs) is accepted by the grammar
pub fn parse(grammar: &Grammar, input: &[u32]) -> Option<ParseTree> {
    let parser = ValiantParser::new(grammar.clone());
    parser.parse_on(input, grammar.start)
}
