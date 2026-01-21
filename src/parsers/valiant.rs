// Valiant parser - equivalent to python_files/final_parser/Valiant.py

use crate::grammars::{Grammar, NumProduction, NumSymbol};
use crate::parse_tree::{ParseSymbol, ParseTree};
use std::collections::{HashMap, HashSet};

pub type NTMatrix = Vec<Vec<HashSet<u32>>>;
pub type BoolMatrix = Vec<Vec<bool>>;
pub type BoolMatrices = HashMap<u32, BoolMatrix>;
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

    /// Compute a hash for a matrix (used as cache key)
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

    /// Convert a matrix of non-terminal sets to boolean matrices (one per non-terminal)
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
    fn multiply_matrices_b(&self, a: &NTMatrix, b: &NTMatrix) -> NTMatrix {
        let length = a.len();
        let bool_as = self.bool_matrices(a);
        let bool_bs = self.bool_matrices(b);
        let r = self.multiply_pairs(&bool_as, &bool_bs);
        self.get_final_matrix(&r, length)
    }

    /// Union of two matrices
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
    // Parse Tree Extraction
    // ========================================================================

    fn find_breaks(
        &self,
        table: &NTMatrix,
        sym: u32,
        start_col: usize,
        end_col: usize,
    ) -> Vec<(usize, u32, u32)> {
        // Find productions sym -> left right
        let productions_for_sym: Vec<_> = self
            .nonterminal_productions
            .iter()
            .filter(|(lhs, _)| *lhs == sym)
            .collect();

        let mut breaks = Vec::new();

        // table[i][j] stores set of nonterminals deriving text[i..j]
        // We are looking for split point k such that:
        // sym -> left right
        // left in table[start_col][k]
        // right in table[k][end_col]

        for k in (start_col + 1)..end_col {
            for &(_, (left_nt, right_nt)) in &productions_for_sym {
                let has_left = table[start_col][k].contains(&left_nt);
                let has_right = table[k][end_col].contains(&right_nt);

                if has_left && has_right {
                    breaks.push((k, *left_nt, *right_nt));
                }
            }
        }

        breaks
    }

    fn extract_tree(
        &self,
        table: &NTMatrix,
        sym: u32,
        text: &[u32],
        start: usize,
        end: usize,
    ) -> Option<ParseTree> {
        let name_str = self.grammar.non_terminal_str(sym).unwrap().to_string();

        // Base case: Leaf node (length 1)
        if end - start == 1 {
            // Check for terminal production sym -> text[start]
            let terminal_val = text[start];
            // Verify this terminal derivation is valid (it should be if passed correctly)
            let is_valid = self
                .terminal_productions
                .iter()
                .any(|(lhs, t)| *lhs == sym && *t == terminal_val);

            if is_valid {
                return Some(ParseTree::new(
                    ParseSymbol::NonTerminal(name_str),
                    vec![ParseTree::new(
                        ParseSymbol::Terminal(
                            self.grammar
                                .terminals
                                .get_str(terminal_val)
                                .unwrap()
                                .to_string(),
                        ),
                        vec![],
                    )],
                ));
            }
        }

        // Recursive step: Find split
        let breaks = self.find_breaks(table, sym, start, end);

        if breaks.is_empty() {
            // Note: This might happen if there's only terminal production but we are length > 1 (impossible in CNF)
            // or if the logic flow is wrong. In strict CNF, length 1 -> terminal, length > 1 -> 2 non-terminals.
            return None;
        }

        // Deterministically pick the first valid break
        let (split, left_nt, right_nt) = breaks[0];

        let left_tree = self.extract_tree(table, left_nt, text, start, split)?;
        let right_tree = self.extract_tree(table, right_nt, text, split, end)?;

        Some(ParseTree {
            name: ParseSymbol::NonTerminal(name_str),
            children: vec![left_tree, right_tree],
        })
    }

    /// Main parsing function
    pub fn parse_on(&mut self, text: &[u32], start_symbol: u32) -> Option<ParseTree> {
        let length = text.len();
        if length == 0 {
            return None;
        }

        // Initialize and fill the base table (terminals)
        let mut table = self.init_nt_matrix(length);
        self.fill_terminals(text, length, &mut table);

        let n = length;
        // Compute transitive closure
        let closure = self.transitive_closure(&table, n);

        if closure[0][n].contains(&start_symbol) {
            return self.extract_tree(&closure, start_symbol, text, 0, n);
        }

        None
    }
}

// Helper requires `Clone` on `ValiantParser` if I use the `..self.clone()` trick,
// but `ValiantParser` fields (`Grammar`, `Vec`) are cloneable.
// Except I haven't derived Clone for ValiantParser.
// I should probably derive Clone or just change `parse_on` to `&mut self`.
// Let's modify `parse_on` to `&mut self` and update the usage in `parse`.

impl Clone for ValiantParser {
    fn clone(&self) -> Self {
        ValiantParser {
            cache: self.cache.clone(),
            cell_width: self.cell_width,
            grammar: self.grammar.clone(),
            productions: self.productions.clone(),
            terminal_productions: self.terminal_productions.clone(),
            nonterminal_productions: self.nonterminal_productions.clone(),
            nonterminals: self.nonterminals.clone(),
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

        let mut parser = ValiantParser::new(cnf);
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

        let mut parser = ValiantParser::new(cnf);
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

        let mut parser = ValiantParser::new(cnf);
        let result = parser.parse_on(&input, parser.grammar.start);

        assert!(result.is_none(), "Should reject 'a a'");
        println!("✓ Correctly rejected invalid input");
    }
}

/// Parse input using Valiant algorithm
/// Returns true if the input (as numeric token IDs) is accepted by the grammar
pub fn parse(grammar: &Grammar, input: &[u32]) -> Option<ParseTree> {
    let mut parser = ValiantParser::new(grammar.clone());
    parser.parse_on(input, grammar.start)
}
