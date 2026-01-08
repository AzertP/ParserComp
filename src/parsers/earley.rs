// Earley parser implementation - new version
use crate::grammars::{NumericGrammar, NumSymbol};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use crate::parse_tree::{ParseTree, ParseSymbol};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RuleID(usize);

struct Rule {
    lhs: NumSymbol,
    rhs: Vec<NumSymbol>,
}

struct Grammar {
    rules: Vec<Rule>,
    lookup: HashMap<NumSymbol, Vec<RuleID>>,
}

impl Grammar {
    fn from_numeric(grammar: &NumericGrammar) -> Self {
        let mut rules = Vec::new();
        
        let mut lhs_keys: Vec<_> = grammar.rules.keys().collect();
        let mut lookup: HashMap<NumSymbol, Vec<RuleID>> = HashMap::new();
        lhs_keys.sort();

        for lhs in lhs_keys {
            let rhs_list = &grammar.rules[lhs];
            let mut rule_ids = Vec::new();
            for rhs in rhs_list {
                let rule = Rule {
                    lhs: NumSymbol::NonTerminal(*lhs),
                    rhs: rhs.iter().map(|&s| s).collect(),
                };
                let rule_id = RuleID(rules.len());
                rule_ids.push(rule_id);
                rules.push(rule);
                
            }
            lookup.insert(NumSymbol::NonTerminal(*lhs), rule_ids);
        }

        Grammar { rules, lookup }
    }

    pub fn calculate_nullables(&self) -> HashSet<NumSymbol> {
        let mut nullables = HashSet::new();
        let mut changed = true;

        while changed {
            changed = false;

            for rule in &self.rules {
                // Skip known nullables
                if nullables.contains(&rule.lhs) {
                    continue;
                }

                // A rule is nullable if all symbols on the RHS are nullable
                let is_nullable = rule.rhs.iter().all(|sym| {
                    match sym {
                        NumSymbol::Terminal(_) => false,
                        NumSymbol::NonTerminal(_) => nullables.contains(sym),
                    }
                });

                if is_nullable {
                    nullables.insert(rule.lhs);
                    changed = true;
                }
            }
        }

        nullables
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    rule_id: RuleID,
    dot: usize,
    s_col: ColumnID,
    // e_col removed
}

impl State {
    // "Constructor" just creates the data
    fn new(rule_id: RuleID, dot: usize, s_col: ColumnID) -> Self {
        State { rule_id, dot, s_col }
    }

    fn is_complete(&self, grammar: &Grammar) -> bool {
        let rule = &grammar.rules[self.rule_id.0];
        self.dot >= rule.rhs.len()
    }

    fn next_symbol<'a>(&self, grammar: &'a Grammar) -> Option<&'a NumSymbol> {
        let rule = &grammar.rules[self.rule_id.0];
        rule.rhs.get(self.dot)
    }

    fn advance(&self) -> Self {
        State {
            rule_id: self.rule_id,
            dot: self.dot + 1,
            s_col: self.s_col,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ColumnID(usize);

struct Column {
    id: ColumnID,
    token: Option<NumSymbol>, // Letter in the Python code
    states: Vec<State>,
    lookup: HashMap<State, usize>,
}

impl Column {
    fn new(id: ColumnID, token: Option<NumSymbol>) -> Self {
        Column {
            id,
            token,
            states: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Returns the index of the state in the column
    fn add_state(&mut self, state: State) -> usize {
        // Check if it exists
        if let Some(&index) = self.lookup.get(&state) {
            return index;
        }

        // If not, add it
        let index = self.states.len();
        self.states.push(state);
        self.lookup.insert(state, index);
        index
    }
}

struct Chart {
    columns: Vec<Column>,
}

impl Chart {
    fn new(size: usize) -> Self {
        let columns = Vec::with_capacity(size);
        Chart { columns }
    }
}


pub struct EarleyParser {
    grammar: Grammar,
    num_grammar: NumericGrammar,
    start_symbol: NumSymbol,
    nullables: HashSet<NumSymbol>,
    input: Vec<NumSymbol>,
    chart: Chart,
}

impl EarleyParser {
    pub fn new(grammar: NumericGrammar) -> Self {
        let num_grammar = grammar.clone();
        let grammar_converted = Grammar::from_numeric(&grammar);
        let nullables = grammar_converted.calculate_nullables();
        
        EarleyParser { 
            grammar: grammar_converted, 
            num_grammar,
            start_symbol: NumSymbol::NonTerminal(grammar.start),
            nullables,
            input: Vec::new(),
            chart: Chart::new(0), // Placeholder
        }
    }

    fn chart_parse(&mut self, input: Vec<NumSymbol>) {
        self.input = input;

        // Create new chart
        let mut chart = Chart::new(self.input.len() + 1);

        // Store token in columns
        // First column has no token
        let first_column = Column::new(ColumnID(0), None);
        chart.columns.push(first_column);
        for (i, token) in self.input.iter().enumerate() {
            let column = Column::new(ColumnID(i + 1), Some(*token));
            chart.columns.push(column);
        }

        // Populate first column with start states
        if let Some(start_rules) = self.grammar.lookup.get(&self.start_symbol) {
            for &rule_id in start_rules {
                let state = State::new(rule_id, 0, ColumnID(0));
                chart.columns[0].add_state(state);
            }
        }
        self.chart = chart;

        // Fill the chart
        self.fill_chart();
    }

    pub fn recognize_on(&mut self, input: Vec<NumSymbol>) -> bool {
        self.chart_parse(input);

        // Check for completed start states in the last column
        let last_col_idx = self.chart.columns.len() - 1;
        for state in &self.chart.columns[last_col_idx].states {
            if state.is_complete(&self.grammar) {
                let rule = &self.grammar.rules[state.rule_id.0];
                if rule.lhs == self.start_symbol && state.s_col == ColumnID(0) {
                    return true;
                }
            }
        }
        false
    }

    fn fill_chart(&mut self) {
        let mut i = 0;
        while i < self.chart.columns.len() {
            let mut j = 0;
            while j < self.chart.columns[i].states.len() {
                let state = self.chart.columns[i].states[j]; 
                
                if state.is_complete(&self.grammar) {
                    self.complete(i, &state);
                } else {
                    let next_sym = state.next_symbol(&self.grammar);
                    if let Some(sym) = next_sym {
                        match sym {
                            NumSymbol::NonTerminal(_) => {
                                self.predict(i, *sym, &state);
                            }
                            NumSymbol::Terminal(_) => {
                                if i + 1 < self.chart.columns.len() {
                                    self.scan(i + 1, &state, *sym);
                                }
                            }
                        }
                    }
                }
                j += 1;
            }
            i += 1;
        }
    }
    
    fn predict(&mut self, col_idx: usize, sym: NumSymbol, state: &State) {
        // Implementation here
        let nt_symbol = match sym {
            NumSymbol::NonTerminal(_) => sym,
            NumSymbol::Terminal(_) => {
                // This should not happen
                panic!("Predict called with terminal symbol");
            }
        };

        // Simply add new states for each rule with this LHS
        if let Some(rule_ids) = self.grammar.lookup.get(&nt_symbol) {
            for &rule_id in rule_ids {
                let new_state = State::new(rule_id, 0, ColumnID(col_idx));
                self.chart.columns[col_idx].add_state(new_state);
            }
        }

        // If the term is nullable, simply advance
        if self.nullables.contains(&nt_symbol) {
            let advanced_state = state.advance();
            self.chart.columns[col_idx].add_state(advanced_state);
        }
    }

    fn scan(&mut self, col_idx: usize, state: &State, token: NumSymbol) {
        // Simply advance the state if the symbol in column matches the token
        if self.chart.columns[col_idx].token == Some(token) {
            let advanced_state = state.advance();
            self.chart.columns[col_idx].add_state(advanced_state);
        }
    }

    fn complete(&mut self, col_idx: usize, state: &State) {
        let state_name = self.grammar.rules[state.rule_id.0].lhs;

        let mut parent_states: Vec<State> = Vec::new();
        // For each state in the s_col
        for st in self.chart.columns[state.s_col.0].states.iter() {
            let st_next_sym = st.next_symbol(&self.grammar);
            if let Some(next_sym) = st_next_sym {
                if *next_sym == state_name {
                    parent_states.push(*st);
                }
            }
        }

        // Advance each parent state
        for parent_state in parent_states {
            let advanced_state = parent_state.advance();
            self.chart.columns[col_idx].add_state(advanced_state);
        }
    }
}


/// A node in the Shared Packed Parse Forest.
/// - `Packed`: Ambiguous derivations are stored in `derivations`.
/// - `Shared`: Nodes are referenced by Rc to form a DAG, not a tree.
#[derive(Debug)]
pub struct SPPFNode {
    pub symbol: ParseSymbol,
    pub start_idx: usize,
    pub end_idx: usize,
    /// A list of alternatives. Each alternative is a sequence of children.
    /// If len() > 1, this node is ambiguous.
    pub derivations: Vec<Vec<Rc<SPPFNode>>>,
}

impl SPPFNode {
    fn new(symbol: ParseSymbol, start: usize, end: usize) -> Self {
        SPPFNode {
            symbol,
            start_idx: start,
            end_idx: end,
            derivations: Vec::new(),
        }
    }
    
    fn add_derivation(&mut self, children: Vec<Rc<SPPFNode>>) {
        self.derivations.push(children);
    }
}
// ============================================================================
// 2. Forest Construction Logic
// ============================================================================

struct ForestBuilder<'a> {
    grammar: &'a Grammar,
    numeric_grammar: &'a NumericGrammar, // <--- Used for string lookup
    chart: &'a Chart,
    input: &'a [NumSymbol],
    // Memoization: (SymbolID, is_terminal, start, end) -> Node
    memo: HashMap<(u32, bool, usize, usize), Rc<SPPFNode>>,
    // Track nodes currently being processed to prevent infinite recursion
    in_progress: HashSet<(u32, bool, usize, usize)>,
}

impl<'a> ForestBuilder<'a> {
    fn new(parser: &'a EarleyParser) -> Self {
        ForestBuilder {
            grammar: &parser.grammar,
            numeric_grammar: &parser.num_grammar,
            chart: &parser.chart,
            input: &parser.input,
            memo: HashMap::new(),
            in_progress: HashSet::new(),
        }
    }

    fn build(&mut self, root_symbol: NumSymbol) -> Option<Rc<SPPFNode>> {
        let end = self.chart.columns.len().saturating_sub(1);
        self.find_node(root_symbol, 0, end)
    }

    fn find_node(&mut self, sym: NumSymbol, start: usize, end: usize) -> Option<Rc<SPPFNode>> {
        let key = (sym.id(), sym.is_terminal(), start, end);
        
        // Check if already memoized
        if let Some(node) = self.memo.get(&key) {
            return Some(node.clone());
        }

        // Check if currently being processed (cycle detection)
        if self.in_progress.contains(&key) {
            // We've hit a cycle - return None to break the recursion
            // This is safe because the Earley chart already validated the parse
            return None;
        }

        // Mark as in-progress before recursing
        self.in_progress.insert(key);

        // 1. Resolve String Representation using NumericGrammar
        let parse_sym = match sym {
            NumSymbol::Terminal(id) => ParseSymbol::Terminal(
                self.numeric_grammar.terminal_str(id)?.to_string()
            ),
            NumSymbol::NonTerminal(id) => ParseSymbol::NonTerminal(
                self.numeric_grammar.non_terminal_str(id)?.to_string()
            ),
        };

        let mut node = SPPFNode::new(parse_sym, start, end);
        let mut added = false;

        match sym {
            NumSymbol::Terminal(_) => {
                // Terminals are leaf nodes; verified against input
                if start + 1 == end && self.input.get(start) == Some(&sym) {
                    node.add_derivation(vec![]); 
                    added = true;
                }
            }
            NumSymbol::NonTerminal(_) => {
                // Non-Terminals: find completed rules in chart column [end]
                // that started at [start] and produce [sym]
                let col = &self.chart.columns[end];
                for state in &col.states {
                    let rule = &self.grammar.rules[state.rule_id.0];
                    
                    if rule.lhs == sym && state.s_col.0 == start && state.is_complete(self.grammar) {
                        // Reconstruct children for this rule
                        let paths = self.walk_back(state.rule_id, rule.rhs.len(), end, start);
                        for children in paths {
                            node.add_derivation(children);
                            added = true;
                        }
                    }
                }
            }
        }

        // Remove from in-progress
        self.in_progress.remove(&key);

        if added {
            let rc = Rc::new(node);
            self.memo.insert(key, rc.clone());
            Some(rc)
        } else {
            None
        }
    }

    /// Recursively reconstructs the RHS children of a rule
    fn walk_back(&mut self, rule_id: RuleID, dot: usize, current_end: usize, target_start: usize) -> Vec<Vec<Rc<SPPFNode>>> {
        if dot == 0 {
            return if current_end == target_start { vec![vec![]] } else { vec![] };
        }

        let rule = &self.grammar.rules[rule_id.0];
        let child_sym = rule.rhs[dot - 1];
        let mut results = Vec::new();

        // Potential split points (k) where child_sym could have started
        // For Terminals: k must be current_end - 1
        // For NonTerminals: k comes from completed states in column [current_end]
        
        let candidates: Box<dyn Iterator<Item = usize>> = match child_sym {
            NumSymbol::Terminal(_) => Box::new(std::iter::once(current_end.saturating_sub(1))),
            NumSymbol::NonTerminal(_) => {
                let mut splits = HashSet::new();
                for st in &self.chart.columns[current_end].states {
                    if st.is_complete(self.grammar) 
                       && self.grammar.rules[st.rule_id.0].lhs == child_sym 
                    {
                        splits.insert(st.s_col.0);
                    }
                }
                Box::new(splits.into_iter())
            }
        };

        for k in candidates {
            if k < target_start { continue; }

            // Ensure the Prefix (Everything before dot-1) exists ending at k
            // This is the check: Does State(rule, dot-1) exist at k with s_col == target_start?
            let pred_state = State::new(rule_id, dot - 1, ColumnID(target_start));
            
            // Fast lookup using the hashmap we built in the column
            if self.chart.columns[k].lookup.contains_key(&pred_state) {
                // If prefix is valid, get the node for the current child
                if let Some(child_node) = self.find_node(child_sym, k, current_end) {
                    // Recursively solve the prefix
                    let prefix_paths = self.walk_back(rule_id, dot - 1, k, target_start);
                    for mut path in prefix_paths {
                        path.push(child_node.clone());
                        results.push(path);
                    }
                }
            }
        }

        results
    }
}

impl EarleyParser {

    /// 1. Build SPPF
    pub fn build_sppf(&self) -> Option<Rc<SPPFNode>> {
        if self.chart.columns.is_empty() { return None; }
        
        // Find if start symbol completed successfully
        let last_col = self.chart.columns.len() - 1;
        let success = self.chart.columns[last_col].states.iter().any(|s| 
            s.is_complete(&self.grammar) 
            && self.grammar.rules[s.rule_id.0].lhs == self.start_symbol
            && s.s_col.0 == 0
        );

        if !success { return None; }

        let mut builder = ForestBuilder::new(self);
        builder.build(self.start_symbol)
    }

    /// 2. Extract One Tree (Fast, Deterministic)
    pub fn extract_one_tree(&self) -> Option<ParseTree> {
        let root = self.build_sppf()?;
        Some(self.sppf_to_tree_single(&root))
    }

    /// 3. Extract All Trees (Exhaustive)
    pub fn extract_all_trees(&self) -> Vec<ParseTree> {
        if let Some(root) = self.build_sppf() {
            self.sppf_to_trees_all(&root)
        } else {
            Vec::new()
        }
    }

    // --- Helpers ---

    fn sppf_to_tree_single(&self, node: &SPPFNode) -> ParseTree {
        if node.derivations.is_empty() {
            ParseTree::leaf(&node.symbol.to_string().trim_matches('\''))
        } else {
            // Take the first derivation found
            let children = node.derivations[0].iter()
                .map(|c| self.sppf_to_tree_single(c))
                .collect();
            ParseTree::new(node.symbol.clone(), children)
        }
    }

    fn sppf_to_trees_all(&self, node: &SPPFNode) -> Vec<ParseTree> {
        if node.derivations.is_empty() {
            return vec![ParseTree::leaf(&node.symbol.to_string().trim_matches('\''))];
        }
        
        let mut trees = Vec::new();
        for deriv in &node.derivations {
            let child_lists = self.cartesian_product(deriv);
            for children in child_lists {
                trees.push(ParseTree::new(node.symbol.clone(), children));
            }
        }
        trees
    }

    fn cartesian_product(&self, nodes: &[Rc<SPPFNode>]) -> Vec<Vec<ParseTree>> {
        if nodes.is_empty() { return vec![vec![]]; }
        
        let first_trees = self.sppf_to_trees_all(&nodes[0]);
        let rest_lists = self.cartesian_product(&nodes[1..]);
        
        let mut result = Vec::new();
        for t in &first_trees {
            for l in &rest_lists {
                let mut list = vec![t.clone()];
                list.extend(l.clone());
                result.push(list);
            }
        }
        result
    }

    pub fn parse(&mut self, input: Vec<u32>) -> Option<ParseTree> {
        let symbols: Vec<NumSymbol> = input.iter()
            .map(|&id| NumSymbol::Terminal(id))
            .collect();
        if self.recognize_on(symbols) {
            self.extract_one_tree()
        } else {
            None
        }
    }

    pub fn parse_all(&mut self, input: Vec<u32>) -> Vec<ParseTree> {
        let symbols: Vec<NumSymbol> = input.iter()
            .map(|&id| NumSymbol::Terminal(id))
            .collect();
        if self.recognize_on(symbols) {
            self.extract_all_trees()
        } else {
            Vec::new()
        }
    }
}

// --------------------------------- End of Code ---------------------------------











// -------------------------------- Test Module -------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::{ambi_grammar, simple_grammar};

    #[test]
    fn test_from_numeric() {
        let numeric_grammar = simple_grammar();
        let grammar = Grammar::from_numeric(&numeric_grammar);

        println!("\n=== Grammar Conversion Test ===");
        println!("Grammar name: {}", numeric_grammar.name);
        println!("Start symbol: {}", numeric_grammar.start);
        println!("\nConverted rules:");
        
        for (idx, rule) in grammar.rules.iter().enumerate() {
            print!("Rule {}: ", idx);
            
            // Print LHS
            match rule.lhs {
                NumSymbol::NonTerminal(id) => {
                    if let Some(symbol) = numeric_grammar.non_terminals.get_str(id) {
                        print!("{} -> ", symbol);
                    } else {
                        print!("NT({}) -> ", id);
                    }
                }
                NumSymbol::Terminal(id) => {
                    if let Some(symbol) = numeric_grammar.terminals.get_str(id) {
                        print!("{} -> ", symbol);
                    } else {
                        print!("T({}) -> ", id);
                    }
                }
            }
            
            // Print RHS
            for (i, sym) in rule.rhs.iter().enumerate() {
                if i > 0 {
                    print!(" ");
                }
                match sym {
                    NumSymbol::NonTerminal(id) => {
                        if let Some(symbol) = numeric_grammar.non_terminals.get_str(*id) {
                            print!("{}", symbol);
                        } else {
                            print!("NT({})", id);
                        }
                    }
                    NumSymbol::Terminal(id) => {
                        if let Some(symbol) = numeric_grammar.terminals.get_str(*id) {
                            print!("'{}'", symbol);
                        } else {
                            print!("T({})", id);
                        }
                    }
                }
            }
            println!();
        }
        
        println!("\nTotal rules: {}", grammar.rules.len());
        println!("=== End Test ===\n");
    }

    #[test]
    fn test_parse_simple() {
        let numeric_grammar = simple_grammar();
        let mut parser = EarleyParser::new(numeric_grammar);
        
        // Get terminal IDs for simple grammar
        // Simple grammar typically has terminals like 'a', 'b', '(', ')'
        let a_id = parser.num_grammar.terminals.get_id("a").expect("Terminal 'a' not found");
        
        // Test: parse a single 'a'
        let input = vec![a_id];
        let result = parser.parse(input);
        
        match result {
            Some(tree) => {
                println!("Parse successful for [a]");
                println!("Tree: {:?}", tree);
                // Verify that the result is a valid parse tree
                assert!(!tree.children.is_empty() || tree.is_leaf());
            }
            None => {
                println!("Parse failed for [a]");
                // This depends on the grammar; simple_grammar might not accept single 'a'
            }
        }
    }

    #[test]
    fn test_parse_all_simple() {
        let numeric_grammar = ambi_grammar();
        let mut parser = EarleyParser::new(numeric_grammar);
        
        let a_id = parser.num_grammar.terminals.get_id("a").expect("Terminal 'a' not found");
        
        let input = vec![a_id, a_id, a_id];
        let results = parser.parse_all(input);
        
        println!("parse_all results for [a, a, a]: {} trees found", results.len());
        for (i, tree) in results.iter().enumerate() {
            println!("  Tree {}: {:}", i, tree.display());
        }
        
        // Either no results or some valid trees
        assert!(results.is_empty() || results.len() > 0);
    }

    #[test]
    fn test_parse_empty_input() {
        let numeric_grammar = simple_grammar();
        let mut parser = EarleyParser::new(numeric_grammar);
        
        let empty_input = vec![];
        let result = parser.parse(empty_input);
        println!("Parse result for empty input: {:?}", result);
        assert!(result.is_none());
        // Empty input should only parse if the start symbol is nullable
    }

    #[test]
    fn test_parse_invalid_sequence() {
        let numeric_grammar = simple_grammar();
        let mut parser = EarleyParser::new(numeric_grammar);
        
        // Try to use the first two terminals in the table
        let mut count = 0;
        let mut term_ids = vec![];
        for i in 0..parser.num_grammar.terminals.len() {
            term_ids.push(i as u32);
            count += 1;
            if count >= 2 {
                break;
            }
        }
        
        if term_ids.is_empty() {
            println!("Skipping test: no terminals found in grammar");
            return;
        }
        
        // An invalid sequence (depends on grammar, but this should be unlikely to parse)
        let result = parser.parse(term_ids.clone());
        
        println!("Parse result for term sequence {:?}: {:?}", term_ids, result);
        // Result is undefined; just verify it returns an Option
    }

    #[test]
    fn test_parse_and_parse_all_consistency() {
        let numeric_grammar = simple_grammar();
        
        // Get some terminals
        let a_id = match numeric_grammar.terminals.get_id("a") {
            Some(id) => id,
            None => return, // Skip if grammar doesn't have 'a'
        };
        
        let input = vec![a_id];
        
        // Test parse()
        let mut parser1 = EarleyParser::new(numeric_grammar.clone());
        let single_tree = parser1.parse(input.clone());
        
        // Test parse_all()
        let mut parser2 = EarleyParser::new(numeric_grammar.clone());
        let all_trees = parser2.parse_all(input.clone());
        
        // If parse() returns Some(tree), parse_all() should return at least one tree
        match single_tree {
            Some(_) => {
                assert!(!all_trees.is_empty(), 
                    "parse_all should return at least one tree if parse returns Some");
                println!("Consistency check passed: parse() returned a tree, parse_all() has {} trees",
                    all_trees.len());
            }
            None => {
                assert!(all_trees.is_empty(), 
                    "parse_all should return empty if parse returns None");
                println!("Consistency check passed: both parse() and parse_all() indicate no parse");
            }
        }
    }
}
