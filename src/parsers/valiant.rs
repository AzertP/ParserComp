// Valiant's recursive closure over a non-associative grammar product.
// The closure/completion recurrence follows Bernardy and Jansson (2016),
// sections 7–8: https://doi.org/10.2168/LMCS-12(2:6)2016.

use crate::grammars::{Grammar, NumProduction, NumSymbol};
use crate::parse_tree::{ParseSymbol, ParseTree};
use rustc_hash::{FxHashMap, FxHashSet};

type HashMap<K, V> = FxHashMap<K, V>;
type HashSet<T> = FxHashSet<T>;

pub type NTMatrix = Vec<Vec<HashSet<u32>>>;

#[derive(Clone, Copy)]
struct Block {
    row: usize,
    col: usize,
    size: usize,
}

impl Block {
    fn new(row: usize, col: usize, size: usize) -> Self {
        Self { row, col, size }
    }
}

/// Square Boolean matrix with packed rows. Multiplication below uses OR/AND,
/// not GF(2): two witnesses must never cancel each other as they would in m4ri.
#[derive(Clone)]
struct BoolMatrix {
    size: usize,
    stride: usize,
    rows: Vec<u64>,
}

impl BoolMatrix {
    fn new(size: usize) -> Self {
        let stride = size.div_ceil(64);
        Self {
            size,
            stride,
            rows: vec![0; size * stride],
        }
    }

    fn set(&mut self, row: usize, col: usize) {
        self.rows[row * self.stride + col / 64] |= 1u64 << (col % 64);
    }

    /// Extract at most log2(size) consecutive bits, including word boundaries.
    fn window(&self, row: usize, start: usize, width: usize) -> usize {
        let word = row * self.stride + start / 64;
        let offset = start % 64;
        let mut bits = self.rows[word] >> offset;
        if offset + width > 64 {
            bits |= self.rows[word + 1] << (64 - offset);
        }
        (bits & ((1u64 << width) - 1)) as usize
    }

    /// Boolean Four Russians multiplication. For each group of k rows of B,
    /// precompute all 2^k subset ORs and look up one subset for every row of A.
    /// k=floor(log2(n)) gives O(n^3/log n) scalar Boolean work (or fewer packed
    /// word operations). There is no fixed cap on k that would lose that bound.
    fn multiply(&self, rhs: &Self) -> Self {
        assert_eq!(self.size, rhs.size);
        let n = self.size;
        let mut result = Self::new(n);
        if n == 0 {
            return result;
        }
        let width = ((usize::BITS - 1 - n.leading_zeros()) as usize).max(1);
        for start in (0..n).step_by(width) {
            let k = width.min(n - start);
            let mut subsets = vec![0u64; (1usize << k) * self.stride];
            for mask in 1usize..(1usize << k) {
                let bit = mask.trailing_zeros() as usize;
                let rest = mask & (mask - 1);
                for word in 0..self.stride {
                    subsets[mask * self.stride + word] = subsets[rest * self.stride + word]
                        | rhs.rows[(start + bit) * self.stride + word];
                }
            }
            for row in 0..n {
                let mask = self.window(row, start, k);
                for word in 0..self.stride {
                    result.rows[row * self.stride + word] |= subsets[mask * self.stride + word];
                }
            }
        }
        result
    }
}

#[derive(Clone)]
pub struct ValiantParser {
    pub cell_width: usize,
    pub grammar: Grammar,
    pub productions: Vec<(u32, NumProduction)>,
    pub terminal_productions: Vec<(u32, u32)>,
    pub nonterminal_productions: Vec<(u32, (u32, u32))>,
    pub nonterminals: Vec<u32>,
    // Multiply only RHS pairs actually used by productions, once per pair.
    binary_rules: Vec<((u32, u32), Vec<u32>)>,
}

impl ValiantParser {
    pub fn new(grammar: Grammar) -> Self {
        let mut productions = Vec::new();
        let mut terminal_productions = Vec::new();
        let mut nonterminal_productions = Vec::new();
        let mut by_rhs: HashMap<(u32, u32), Vec<u32>> = HashMap::default();
        let mut nonterminals: Vec<u32> = grammar.rules.keys().copied().collect();
        nonterminals.sort_unstable();
        for (&lhs, rhs_list) in &grammar.rules {
            for rhs in rhs_list {
                productions.push((lhs, rhs.clone()));
                match rhs.as_slice() {
                    [NumSymbol::Terminal(t)] => terminal_productions.push((lhs, *t)),
                    [NumSymbol::NonTerminal(b), NumSymbol::NonTerminal(c)] => {
                        nonterminal_productions.push((lhs, (*b, *c)));
                        by_rhs.entry((*b, *c)).or_default().push(lhs);
                    }
                    _ => {} // CNF's nullable start is handled separately.
                }
            }
        }
        let mut binary_rules: Vec<_> = by_rhs.into_iter().collect();
        binary_rules.sort_unstable_by_key(|(rhs, _)| *rhs);
        Self {
            cell_width: 5,
            grammar,
            productions,
            terminal_productions,
            nonterminal_productions,
            nonterminals,
            binary_rules,
        }
    }

    fn bool_matrices(table: &NTMatrix, block: Block) -> HashMap<u32, BoolMatrix> {
        let mut matrices: HashMap<u32, BoolMatrix> = HashMap::default();
        for row in 0..block.size {
            for col in 0..block.size {
                for &nt in &table[block.row + row][block.col + col] {
                    matrices
                        .entry(nt)
                        .or_insert_with(|| BoolMatrix::new(block.size))
                        .set(row, col);
                }
            }
        }
        matrices
    }

    /// C += A * B under the grammar product. Sources are disjoint from C in
    /// the completion recurrence; temporary Boolean products preserve them.
    fn add_product(&self, table: &mut NTMatrix, c: Block, a: Block, b: Block) {
        debug_assert_eq!(c.size, a.size);
        debug_assert_eq!(c.size, b.size);
        #[cfg(test)]
        PRODUCT_CELLS.with(|count| count.set(count.get() + c.size * c.size));
        if c.size == 1 {
            let mut parents = Vec::new();
            for &((left, right), ref lhs) in &self.binary_rules {
                if table[a.row][a.col].contains(&left) && table[b.row][b.col].contains(&right) {
                    parents.extend(lhs.iter().copied());
                }
            }
            table[c.row][c.col].extend(parents);
            return;
        }
        let left = Self::bool_matrices(table, a);
        let right = Self::bool_matrices(table, b);
        for &((nt_b, nt_c), ref parents) in &self.binary_rules {
            if let (Some(lhs), Some(rhs)) = (left.get(&nt_b), right.get(&nt_c)) {
                let product = lhs.multiply(rhs);
                for row in 0..c.size {
                    for word in 0..product.stride {
                        let mut bits = product.rows[row * product.stride + word];
                        while bits != 0 {
                            let col = word * 64 + bits.trailing_zeros() as usize;
                            table[c.row + row][c.col + col].extend(parents.iter().copied());
                            bits &= bits - 1;
                        }
                    }
                }
            }
        }
    }

    /// Solve X = Y ∪ A*X ∪ X*B in place, with closed strictly upper triangular
    /// A=table[left,left], B=table[right,right], and Y=table[left,right].
    /// The four smaller X blocks depend on one another in this order:
    /// X21, X11, X22, X12. No associativity of the grammar product is assumed.
    fn complete(&self, table: &mut NTMatrix, left: usize, right: usize, size: usize) {
        if size <= 1 {
            return; // The diagonal entries of A and B are empty.
        }
        let h = size / 2;
        let a12 = Block::new(left, left + h, h);
        let b12 = Block::new(right, right + h, h);
        let x11 = Block::new(left, right, h);
        let x12 = Block::new(left, right + h, h);
        let x21 = Block::new(left + h, right, h);
        let x22 = Block::new(left + h, right + h, h);

        self.complete(table, left + h, right, h);
        self.add_product(table, x11, a12, x21);
        self.complete(table, left, right, h);
        self.add_product(table, x22, x21, b12);
        self.complete(table, left + h, right + h, h);
        self.add_product(table, x12, a12, x22);
        self.add_product(table, x12, x11, b12);
        self.complete(table, left, right + h, h);
    }

    fn close(&self, table: &mut NTMatrix, start: usize, size: usize) {
        if size <= 1 {
            return;
        }
        let h = size / 2;
        self.close(table, start, h);
        self.close(table, start + h, h);
        self.complete(table, start, start + h, h);
    }

    /// Pad position count (not token count) to a power of two. Padding has no
    /// terminal edges and cannot derive additional spans in the real input.
    fn recognition_chart(&self, text: &[u32]) -> NTMatrix {
        let size = (text.len() + 1).next_power_of_two();
        let mut table = vec![vec![HashSet::default(); size]; size];
        for (pos, &token) in text.iter().enumerate() {
            for &(lhs, terminal) in &self.terminal_productions {
                if token == terminal {
                    table[pos][pos + 1].insert(lhs);
                }
            }
        }
        self.close(&mut table, 0, size);
        table
    }

    pub fn recognize_on(&mut self, text: &[u32], start_symbol: u32) -> bool {
        if text.is_empty() {
            return start_symbol == self.grammar.start && self.grammar.start_nullable;
        }
        self.recognition_chart(text)[0][text.len()].contains(&start_symbol)
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
            // Valiant's algorithm requires n ≥ 1; use `start_nullable` for ε.
            if start_symbol == self.grammar.start && self.grammar.start_nullable {
                return Some(ParseTree::new(
                    ParseSymbol::NonTerminal(
                        self.grammar
                            .non_terminal_str(start_symbol)
                            .unwrap_or("S")
                            .to_string(),
                    ),
                    vec![],
                ));
            }
            return None;
        }

        let n = length;
        let closure = self.recognition_chart(text);

        if closure[0][n].contains(&start_symbol) {
            return self.extract_tree(&closure, start_symbol, text, 0, n);
        }

        None
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
/// Returns one parse tree if the numeric token sequence is accepted.
pub fn parse(grammar: &Grammar, input: &[u32]) -> Option<ParseTree> {
    let mut parser = ValiantParser::new(grammar.clone());
    parser.parse_on(input, grammar.start)
}

#[cfg(test)]
thread_local! {
    // Count matrix cells processed by grammar-matrix products, independently
    // of CPU speed, to catch reintroduction of the quartic conversion work.
    static PRODUCT_CELLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::grammars::load_grammar_from_str;

    fn ambiguous_parser() -> ValiantParser {
        let grammar = load_grammar_from_str(
            r#"{
            "name": "regression", "start": "<S>", "rules": {"<S>": [["<S>", "<S>"], ["a"]]}
        }"#,
        )
        .unwrap();
        ValiantParser::new(grammar.to_cnf())
    }

    #[test]
    fn boolean_product_preserves_multiple_witnesses() {
        let mut a = BoolMatrix::new(2);
        let mut b = BoolMatrix::new(2);
        a.set(0, 0);
        a.set(0, 1);
        b.set(0, 0);
        b.set(1, 0);
        let product = a.multiply(&b);
        assert_eq!(product.rows, vec![1, 0]);
    }

    #[test]
    fn recognition_accepts_nullable_start_like_parsing() {
        let grammar = load_grammar_from_str(
            r#"{
            "name": "regression", "start": "<S>", "rules": {"<S>": [[], ["a"]]}
        }"#,
        )
        .unwrap()
        .to_cnf();
        let mut parser = ValiantParser::new(grammar);
        let start = parser.grammar.start;
        assert!(parser.parse_on(&[], start).is_some());
        assert!(parser.recognize_on(&[], start));
    }

    #[test]
    fn closure_avoids_quartic_matrix_conversion_work() {
        let mut work = Vec::new();
        for n in [7, 15] {
            let mut parser = ambiguous_parser();
            let tokens = parser.grammar.tokenize(&"a".repeat(n)).unwrap();
            PRODUCT_CELLS.with(|count| count.set(0));
            assert!(parser.recognize_on(&tokens, parser.grammar.start));
            work.push(PRODUCT_CELLS.with(|count| count.get()));
        }
        assert!(work[0] > 0);
        assert!(
            work[1] <= 8 * work[0],
            "doubling the chart must not quadruple both product count and product size: {work:?}"
        );
    }

    #[test]
    fn boolean_product_matches_scalar_oracle_across_word_boundaries() {
        let mut seed = 19u64;
        for n in [0, 1, 2, 3, 15, 16, 31, 63, 64, 65, 127, 129] {
            for density in [1, 4, 15] {
                let mut a = BoolMatrix::new(n);
                let mut b = BoolMatrix::new(n);
                let mut left = vec![vec![false; n]; n];
                let mut right = left.clone();
                for i in 0..n {
                    for j in 0..n {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        left[i][j] = seed >> 60 < density;
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        right[i][j] = seed >> 60 < density;
                        if left[i][j] {
                            a.set(i, j);
                        }
                        if right[i][j] {
                            b.set(i, j);
                        }
                    }
                }
                let actual = a.multiply(&b);
                let mut expected = BoolMatrix::new(n);
                for i in 0..n {
                    for j in 0..n {
                        if (0..n).any(|k| left[i][k] && right[k][j]) {
                            expected.set(i, j);
                        }
                    }
                }
                assert_eq!(actual.rows, expected.rows, "n={n}, density={density}");
            }
        }
    }

    // Independent bottom-up CYK oracle. Deliberately reads grammar rules,
    // rather than using the implementation's precomputed production indexes.
    fn cyk_chart(grammar: &Grammar, tokens: &[u32]) -> NTMatrix {
        let mut chart = vec![vec![HashSet::default(); tokens.len() + 1]; tokens.len() + 1];
        for span in 1..=tokens.len() {
            for i in 0..=tokens.len() - span {
                let j = i + span;
                for (&lhs, alternatives) in &grammar.rules {
                    for rhs in alternatives {
                        let derives = match rhs.as_slice() {
                            [NumSymbol::Terminal(t)] => span == 1 && tokens[i] == *t,
                            [NumSymbol::NonTerminal(b), NumSymbol::NonTerminal(c)] => (i + 1..j)
                                .any(|k| chart[i][k].contains(b) && chart[k][j].contains(c)),
                            _ => false,
                        };
                        if derives {
                            chart[i][j].insert(lhs);
                        }
                    }
                }
            }
        }
        chart
    }

    fn check_tree(grammar: &Grammar, tree: &ParseTree) {
        let ParseSymbol::NonTerminal(ref name) = tree.name else {
            panic!("expected nonterminal");
        };
        let lhs = grammar
            .rules
            .keys()
            .copied()
            .find(|&nt| grammar.non_terminal_str(nt) == Some(name.as_str()))
            .unwrap();
        if tree.children.is_empty() && lhs == grammar.start && grammar.start_nullable {
            return;
        }
        let production: Vec<_> = tree
            .children
            .iter()
            .map(|child| match &child.name {
                ParseSymbol::Terminal(t) => {
                    assert!(child.children.is_empty());
                    NumSymbol::Terminal(grammar.terminals.get_id(t).unwrap())
                }
                ParseSymbol::NonTerminal(n) => {
                    check_tree(grammar, child);
                    NumSymbol::NonTerminal(
                        grammar
                            .rules
                            .keys()
                            .copied()
                            .find(|&nt| grammar.non_terminal_str(nt) == Some(n.as_str()))
                            .unwrap(),
                    )
                }
            })
            .collect();
        assert!(
            grammar.rules[&lhs].contains(&production),
            "invalid tree rule: {tree:?}"
        );
    }

    fn check_input(parser: &mut ValiantParser, input: &str) {
        let tokens = parser.grammar.tokenize(input).unwrap();
        let expected = cyk_chart(&parser.grammar, &tokens);
        let actual = parser.recognition_chart(&tokens);
        for i in 0..actual.len() {
            for j in 0..actual.len() {
                if i <= tokens.len() && j <= tokens.len() {
                    assert_eq!(
                        actual[i][j], expected[i][j],
                        "input={input:?}, cell=({i},{j})"
                    );
                } else {
                    assert!(actual[i][j].is_empty(), "padding must stay empty");
                }
            }
        }
        let start = parser.grammar.start;
        let accepted = if tokens.is_empty() {
            parser.grammar.start_nullable
        } else {
            expected[0][tokens.len()].contains(&start)
        };
        assert_eq!(parser.recognize_on(&tokens, start), accepted);
        let tree = parser.parse_on(&tokens, start);
        assert_eq!(tree.is_some(), accepted);
        if let Some(tree) = tree {
            assert_eq!(tree.to_flat_string(), input);
            check_tree(&parser.grammar, &tree);
        }
    }

    #[test]
    fn every_chart_cell_matches_cyk_on_exhaustive_inputs() {
        let rules = [
            r#"{"<S>":[["<S>","<S>"],["a"],["b"]]}"#,
            r#"{"<S>":[["<A>","<B>"],["<C>","<D>"]],"<A>":[["<C>","<D>"],["a"]],"<B>":[["<D>","<C>"],["b"]],"<C>":[["a"]],"<D>":[["b"]]}"#,
            r#"{"<S>":[[],["<A>"]],"<A>":[["<S>"],["a","<S>","b"]]}"#,
        ];
        for rules in rules {
            let json = format!(r#"{{"name":"oracle","start":"<S>","rules":{rules}}}"#);
            let mut parser = ValiantParser::new(load_grammar_from_str(&json).unwrap().to_cnf());
            for n in 0..=6 {
                for mask in 0..1usize << n {
                    let input: String = (0..n)
                        .map(|i| if mask & (1 << i) == 0 { 'a' } else { 'b' })
                        .collect();
                    check_input(&mut parser, &input);
                }
            }
        }
    }

    #[test]
    fn closure_matches_cyk_at_padding_and_packed_word_boundaries() {
        let mut parser = ambiguous_parser();
        for n in [7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129] {
            check_input(&mut parser, &"a".repeat(n));
        }
    }
}
