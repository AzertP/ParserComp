// CYK parser - equivalent to python_files/final_parser/CYK.py

use crate::grammars::{NumProduction, NumSymbol, NumericGrammar};
use crate::parse_tree::{ParseSymbol, ParseTree};
use rustc_hash::FxHashMap;

type HashMap<K, V> = FxHashMap<K, V>;

/// A forest node represents multiple possible parse trees
/// Each entry is (non-terminal, children) where children are references to other forest nodes
pub type ForestNode = Vec<(u32, Vec<ForestRef>)>;

/// Reference to a forest node in the table, or a terminal leaf
#[derive(Clone, Debug)]
pub enum ForestRef {
    /// Reference to table[row][col][nt]
    TableRef { row: usize, col: usize, nt: u32 },
    /// Terminal leaf node
    Terminal(u32),
}

/// CYK Parse Table: table[s][e] maps non-terminal -> list of (nt, children) derivations
pub type CYKTable = Vec<Vec<HashMap<u32, ForestNode>>>;

/// One witness for a nonterminal's derivation of a span.
#[derive(Clone, Copy, Debug)]
enum Backpointer {
    Terminal(u32),
    Binary { split: usize, left: u32, right: u32 },
}

type OneTreeTable = Vec<Vec<HashMap<u32, Backpointer>>>;

pub struct CYKParser {
    pub cell_width: usize,
    pub grammar: NumericGrammar,
    pub productions: Vec<(u32, NumProduction)>,
    pub terminal_productions: Vec<(u32, u32)>,
    pub nonterminal_productions: Vec<(u32, (u32, u32))>,
}

impl CYKParser {
    pub fn new(grammar: NumericGrammar) -> Self {
        let mut productions: Vec<(u32, NumProduction)> = Vec::new();
        let mut terminal_productions: Vec<(u32, u32)> = Vec::new();
        let mut nonterminal_productions: Vec<(u32, (u32, u32))> = Vec::new();

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

        CYKParser {
            cell_width: 5,
            grammar,
            productions,
            terminal_productions,
            nonterminal_productions,
        }
    }

    /// Initialize the parse table
    /// table[s][e] = HashMap from non-terminal to list of derivations
    fn init_table(&self, length: usize) -> CYKTable {
        // We need table[s][e] for 0 <= s <= e <= length
        // Using length+1 for both dimensions
        vec![vec![HashMap::default(); length + 1]; length + 1]
    }

    /// Parse terminals (length 1 substrings)
    fn parse_1(&self, text: &[u32], length: usize, table: &mut CYKTable) {
        for s in 0..length {
            for &(key, terminal) in &self.terminal_productions {
                if text[s] == terminal {
                    let entry = table[s][s + 1].entry(key).or_insert_with(Vec::new);
                    entry.push((key, vec![ForestRef::Terminal(text[s])]));
                }
            }
        }
    }

    /// Parse non-terminals for spans of length n
    /// Equivalent to parse_n in Python
    fn parse_n(&self, n: usize, length: usize, table: &mut CYKTable) {
        for s in 0..=length - n {
            for p in 1..n {
                for &(k, (r_b, r_c)) in &self.nonterminal_productions {
                    // Check if R_b is in table[s][s+p] and R_c is in table[s+p][s+n]
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

    /// Extract a parse tree from a forest node
    /// Equivalent to trees() in Python - picks first derivation (deterministic)
    fn trees(&self, table: &CYKTable, forest_ref: &ForestRef) -> Option<ParseTree> {
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

                // Pick first derivation (like random.choice but deterministic)
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

    /// Build a CNF chart retaining the first derivation for each nonterminal
    /// in each span. Other nonterminals in that span must still be retained.
    fn one_tree_chart(&self, text: &[u32]) -> OneTreeTable {
        let length = text.len();
        let mut table = vec![vec![HashMap::default(); length + 1]; length + 1];
        for (s, &token) in text.iter().enumerate() {
            for &(lhs, terminal) in &self.terminal_productions {
                if token == terminal {
                    table[s][s + 1]
                        .entry(lhs)
                        .or_insert(Backpointer::Terminal(token));
                }
            }
        }
        for span in 2..=length {
            for s in 0..=length - span {
                let end = s + span;
                for split in s + 1..end {
                    for &(lhs, (left, right)) in &self.nonterminal_productions {
                        if table[s][split].contains_key(&left)
                            && table[split][end].contains_key(&right)
                        {
                            table[s][end].entry(lhs).or_insert(Backpointer::Binary {
                                split,
                                left,
                                right,
                            });
                        }
                    }
                }
            }
        }
        table
    }

    fn one_tree(
        &self,
        table: &OneTreeTable,
        start: usize,
        end: usize,
        nt: u32,
    ) -> Option<ParseTree> {
        let children = match *table[start][end].get(&nt)? {
            Backpointer::Terminal(token) => vec![ParseTree::new(
                ParseSymbol::Terminal(self.grammar.terminals.get_str(token)?.to_string()),
                vec![],
            )],
            Backpointer::Binary { split, left, right } => vec![
                self.one_tree(table, start, split, left)?,
                self.one_tree(table, split, end, right)?,
            ],
        };
        Some(ParseTree::new(
            ParseSymbol::NonTerminal(self.grammar.non_terminal_str(nt)?.to_string()),
            children,
        ))
    }

    /// Parse a CNF token sequence, retaining one backpointer per span and
    /// nonterminal instead of a forest of alternative derivations.
    /// Returns the first tree in the same traversal order as `parse_on`.
    pub fn parse_one_on(&self, text: &[u32], start_symbol: u32) -> Option<ParseTree> {
        if text.is_empty() {
            return (start_symbol == self.grammar.start && self.grammar.start_nullable).then(
                || {
                    ParseTree::new(
                        ParseSymbol::NonTerminal(
                            self.grammar.start_str().unwrap_or("S").to_string(),
                        ),
                        vec![],
                    )
                },
            );
        }
        let table = self.one_tree_chart(text);
        self.one_tree(&table, 0, text.len(), start_symbol)
    }

    /// Main parsing function
    /// Returns a parse tree if successful, None otherwise
    pub fn parse_on(&self, text: &[u32], start_symbol: u32) -> Option<ParseTree> {
        let length = text.len();
        if length == 0 {
            // Standard CYK cannot handle ε; defer to the `start_nullable` flag
            // that `to_cnf()` sets when the original start symbol derives ε.
            if self.grammar.start_nullable {
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

        let mut table = self.init_table(length);
        self.parse_1(text, length, &mut table);

        for n in 2..=length {
            self.parse_n(n, length, &mut table);
        }

        // Get the forest for start_symbol at table[0][length]
        if let Some(forest) = table[0][length].get(&start_symbol) {
            if !forest.is_empty() {
                let root_ref = ForestRef::TableRef {
                    row: 0,
                    col: length,
                    nt: start_symbol,
                };
                return self.trees(&table, &root_ref);
            }
        }

        None
    }
}

/// Parse input using CYK algorithm (recognizer only)
/// Returns true if the input is accepted by the grammar
pub fn recognize(grammar: &NumericGrammar, input: &[u32]) -> bool {
    let cnf = grammar.to_cnf();
    let parser = CYKParser::new(cnf);
    parser.parse_on(input, parser.grammar.start).is_some()
}

/// Parse input using CYK algorithm
/// Returns a parse tree if successful
pub fn parse(grammar: &NumericGrammar, input: &[u32]) -> Option<ParseTree> {
    // Grammar should already be in CNF
    let parser: CYKParser = CYKParser::new(grammar.clone());
    return parser.parse_on(input, parser.grammar.start);
}

/// Parse an already-CNF grammar, retaining one backpointer per span/nonterminal.
/// Parser construction and tree extraction are included in this call.
pub fn parse_one(grammar: &NumericGrammar, input: &[u32]) -> Option<ParseTree> {
    let parser = CYKParser::new(grammar.clone());
    parser.parse_one_on(input, parser.grammar.start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::load_grammar_from_str;

    #[test]
    fn test_cyk_simple_grammar() {
        // Simple arithmetic grammar:
        // E -> E + T | T
        // T -> T * F | F
        // F -> ( E ) | a
        //
        // But for CYK we need CNF, so let's use a simpler example:
        // S -> A B
        // A -> a
        // B -> b
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

        println!("=== Testing CYK Parser ===");
        println!("Original grammar terminals: {:?}", grammar.terminals);
        println!(
            "Original grammar non-terminals: {:?}",
            grammar.non_terminals
        );
        println!(
            "Original grammar start: {} -> {:?}",
            grammar.start,
            grammar.start_str()
        );

        // Convert to CNF and check
        let cnf = grammar.to_cnf();
        println!("\nCNF grammar terminals: {:?}", cnf.terminals);
        println!("CNF grammar non-terminals: {:?}", cnf.non_terminals);
        println!("CNF grammar start: {} -> {:?}", cnf.start, cnf.start_str());
        println!("CNF grammar rules:");
        for (&nt, prods) in &cnf.rules {
            let nt_name = cnf.non_terminal_str(nt).unwrap_or("?");
            for prod in prods {
                let prod_str: Vec<String> = prod
                    .iter()
                    .map(|s| match s {
                        NumSymbol::Terminal(t) => {
                            format!("'{}'", cnf.terminal_str(*t).unwrap_or("?"))
                        }
                        NumSymbol::NonTerminal(n) => {
                            cnf.non_terminal_str(*n).unwrap_or("?").to_string()
                        }
                    })
                    .collect();
                println!("  {} -> {}", nt_name, prod_str.join(" "));
            }
        }

        let cnf = grammar.to_cnf();

        // Tokenize "a b" -> get token IDs from CNF grammar
        let token_a = cnf.terminals.get_id("a").expect("Token 'a' not found");
        let token_b = cnf.terminals.get_id("b").expect("Token 'b' not found");
        let input = vec![token_a, token_b];

        println!("\nInput tokens: {:?} (a={}, b={})", input, token_a, token_b);

        let parser = CYKParser::new(cnf);
        let result = parser.parse_on(&input, parser.grammar.start);

        match &result {
            Some(tree) => {
                println!("\n✓ Parse successful!");
                println!("\nParse tree (raw): {:?}", tree);
                println!("\nParse tree (pretty):");
                println!("{}", tree.display());
            }
            None => {
                println!("\n✗ Parse failed!");
            }
        }

        assert!(result.is_some(), "Should parse 'a b' successfully");
    }

    #[test]
    fn test_cyk_longer_input() {
        // Grammar: S -> A B | S S
        // A -> a
        // B -> b
        // This can parse: "a b", "a b a b", etc.
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

        // Convert to CNF first, then get token IDs from CNF grammar
        let cnf = grammar.to_cnf();
        let token_a = cnf.terminals.get_id("a").expect("Token 'a' not found");
        let token_b = cnf.terminals.get_id("b").expect("Token 'b' not found");
        // Test "a b a b"
        let input = vec![token_a, token_b, token_a, token_b];

        println!("\n=== Testing CYK Parser (longer input) ===");
        println!("Input tokens: {:?}", input);

        let parser = CYKParser::new(cnf);
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
    fn test_cyk_reject_invalid() {
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

        let token_a = grammar.terminals.get_id("a").expect("Token 'a' not found");

        // Test "a a" - should fail (expects "a b")
        let input = vec![token_a, token_a];

        println!("\n=== Testing CYK Parser (invalid input) ===");
        println!("Input tokens: {:?} (should be rejected)", input);

        let result = parse(&grammar, &input);

        assert!(result.is_none(), "Should reject 'a a'");
        println!("✓ Correctly rejected invalid input");
    }
}

#[cfg(test)]
mod one_tree_tests {
    use super::*;
    use crate::grammars::load_grammar_from_str;

    fn parser(rules: &str) -> CYKParser {
        let json = format!(r#"{{"name":"one-tree","start":"<S>","rules":{rules}}}"#);
        CYKParser::new(load_grammar_from_str(&json).unwrap().to_cnf())
    }

    fn check_tree(grammar: &NumericGrammar, tree: &ParseTree) {
        let ParseSymbol::NonTerminal(ref name) = tree.name else {
            panic!("expected nonterminal node");
        };
        let lhs = grammar
            .rules
            .keys()
            .copied()
            .find(|&nt| grammar.non_terminal_str(nt) == Some(name.as_str()))
            .unwrap();
        if tree.children.is_empty() {
            assert_eq!(lhs, grammar.start);
            assert!(grammar.start_nullable);
            return;
        }
        let rhs: Vec<_> = tree
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
            grammar.rules[&lhs].contains(&rhs),
            "invalid production: {tree:?}"
        );
    }

    #[test]
    fn one_tree_matches_forest_chart_and_first_tree() {
        for rules in [
            r#"{"<S>":[["<S>","<S>"],["a"],["b"]]}"#,
            r#"{"<S>":[["<A>","<B>"],["<B>","<A>"]],"<A>":[["a"],["<A>","<A>"]],"<B>":[["b"],["<B>","<B>"]]}"#,
            r#"{"<S>":[[],["<A>"]],"<A>":[["<S>"],["a","<S>","b"]]}"#,
        ] {
            let parser = parser(rules);
            for n in 0..=6 {
                for mask in 0..1usize << n {
                    let input: String = (0..n)
                        .map(|i| if mask & (1 << i) == 0 { 'a' } else { 'b' })
                        .collect();
                    let tokens = parser.grammar.tokenize(&input).unwrap();
                    let single = parser.one_tree_chart(&tokens);
                    let mut forest = parser.init_table(n);
                    parser.parse_1(&tokens, n, &mut forest);
                    for span in 2..=n {
                        parser.parse_n(span, n, &mut forest);
                    }
                    for s in 0..=n {
                        for end in 0..=n {
                            assert_eq!(single[s][end].len(), forest[s][end].len());
                            for nt in forest[s][end].keys() {
                                assert!(
                                    single[s][end].contains_key(nt),
                                    "missing NT at {s}..{end}"
                                );
                            }
                        }
                    }
                    let start = parser.grammar.start;
                    let tree = parser.parse_one_on(&tokens, start);
                    assert_eq!(tree, parser.parse_on(&tokens, start), "input {input:?}");
                    assert_eq!(tree, parse_one(&parser.grammar, &tokens));
                    if let Some(tree) = tree {
                        assert_eq!(tree.to_flat_string(), input);
                        check_tree(&parser.grammar, &tree);
                    }
                }
            }
        }
    }

    #[test]
    fn ambiguous_span_keeps_one_backpointer_per_nonterminal() {
        let parser =
            parser(r#"{"<S>":[["<S>","<S>"],["<A>","<A>"],["a"]],"<A>":[["<A>","<A>"],["a"]]}"#);
        let tokens = parser.grammar.tokenize("aaaa").unwrap();
        let mut forest = parser.init_table(4);
        parser.parse_1(&tokens, 4, &mut forest);
        for n in 2..=4 {
            parser.parse_n(n, 4, &mut forest);
        }
        assert!(forest[0][4][&parser.grammar.start].len() > 1);
        let single = parser.one_tree_chart(&tokens);
        assert_eq!(single[0][4].len(), forest[0][4].len());
        assert!(single[0][4].len() > 1, "retain all deriving nonterminals");
        assert!(matches!(
            single[0][4][&parser.grammar.start],
            Backpointer::Binary { split: 1, .. }
        ));
    }

    #[test]
    fn one_tree_handles_empty_unknown_and_requested_start() {
        let parser = parser(r#"{"<S>":[[],["<A>","<A>"]],"<A>":[["a"]]}"#);
        assert!(parser.parse_one_on(&[], parser.grammar.start).is_some());
        let other = *parser
            .grammar
            .rules
            .keys()
            .find(|&&nt| parser.grammar.non_terminal_str(nt) == Some("<A>"))
            .unwrap();
        assert!(parser.parse_one_on(&[], other).is_none());
        let tokens = parser.grammar.tokenize("a").unwrap();
        assert!(parser.parse_one_on(&tokens, other).is_some());
        assert!(parser.parse_one_on(&tokens, parser.grammar.start).is_none());
        assert!(parser
            .parse_one_on(&[u32::MAX], parser.grammar.start)
            .is_none());
        assert!(parser.parse_one_on(&tokens, u32::MAX).is_none());
        let nonnullable = self::parser(r#"{"<S>":[["a"]]}"#);
        assert!(nonnullable
            .parse_one_on(&[], nonnullable.grammar.start)
            .is_none());
    }
}
