// Earley parser implementation with Leo Optimizations
#[cfg(test)]
use crate::grammars::load_grammar_from_str;
use crate::grammars::{NumSymbol, NumericGrammar};
use crate::parse_tree::{ParseSymbol, ParseTree};
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

type HashMap<K, V> = FxHashMap<K, V>;
type HashSet<T> = FxHashSet<T>;

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
        let mut lookup: HashMap<NumSymbol, Vec<RuleID>> = HashMap::default();
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
        let mut nullables = HashSet::default();
        let mut changed = true;

        while changed {
            changed = false;

            for rule in &self.rules {
                if nullables.contains(&rule.lhs) {
                    continue;
                }

                let is_nullable = rule.rhs.iter().all(|sym| match sym {
                    NumSymbol::Terminal(_) => false,
                    NumSymbol::NonTerminal(_) => nullables.contains(sym),
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
}

impl State {
    fn new(rule_id: RuleID, dot: usize, s_col: ColumnID) -> Self {
        State {
            rule_id,
            dot,
            s_col,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ColumnID(usize);

struct Column {
    _id: ColumnID,
    token: Option<NumSymbol>,
    states: Vec<State>,
    lookup: HashMap<State, usize>,
    /// Transitive items for Leo optimization: Maps (Symbol) -> State (The Top State)
    /// Used to memoize the top of a right-recursive chain.
    transitives: HashMap<NumSymbol, State>,
}

impl Column {
    fn new(id: ColumnID, token: Option<NumSymbol>) -> Self {
        Column {
            _id: id,
            token,
            states: Vec::new(),
            lookup: HashMap::default(),
            transitives: HashMap::default(),
        }
    }

    fn add_state(&mut self, state: State) -> bool {
        use std::collections::hash_map::Entry;
        match self.lookup.entry(state) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                let index = self.states.len();
                self.states.push(state);
                e.insert(index);
                true
            }
        }
    }

    fn add_transitive(&mut self, symbol: NumSymbol, state: State) {
        // In Python's Leo logic, we store the resulting TOP state mapped by the symbol that triggered it.
        self.transitives.insert(symbol, state);
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

pub struct LeoParser {
    grammar: Grammar,
    num_grammar: NumericGrammar,
    start_symbol: NumSymbol,
    nullables: HashSet<NumSymbol>,
    input: Vec<NumSymbol>,
    chart: Chart,

    // Leo Optimization: _postdots
    // Tracks the parent-child relationship in deterministic reductions.
    // Key: The Parent State (at the dot before reduction)
    // Value: List of Child States (completed) that reduced to this parent, along with their end column.
    postdots: HashMap<State, Vec<(State, ColumnID)>>,

    // Reverse Index: Maps a State to the list of Column IDs (indices) where it appears.
    // Used for efficient Implicit Reconstruction lookup.
    state_to_cols: HashMap<State, Vec<usize>>,
}

impl LeoParser {
    pub fn new(grammar: NumericGrammar) -> Self {
        let num_grammar = grammar.clone();
        let grammar_converted = Grammar::from_numeric(&grammar);
        let nullables = grammar_converted.calculate_nullables();

        LeoParser {
            grammar: grammar_converted,
            num_grammar,
            start_symbol: NumSymbol::NonTerminal(grammar.start),
            nullables,
            input: Vec::new(),
            chart: Chart::new(0),
            postdots: HashMap::default(),
            state_to_cols: HashMap::default(),
        }
    }

    fn chart_parse(&mut self, input: Vec<NumSymbol>) {
        self.input = input;
        self.postdots.clear(); // Reset optimizations for new parse

        let mut chart = Chart::new(self.input.len() + 1);

        // Initialize Column 0
        let first_column = Column::new(ColumnID(0), None);
        chart.columns.push(first_column);

        // Initialize other columns
        for (i, token) in self.input.iter().enumerate() {
            let column = Column::new(ColumnID(i + 1), Some(*token));
            chart.columns.push(column);
        }

        // Seed Column 0
        if let Some(start_rules) = self.grammar.lookup.get(&self.start_symbol) {
            for &rule_id in start_rules {
                let state = State::new(rule_id, 0, ColumnID(0));
                chart.columns[0].add_state(state);
            }
        }
        self.chart = chart;

        self.fill_chart();
    }

    pub fn recognize_on(&mut self, input: Vec<NumSymbol>) -> bool {
        self.chart_parse(input);

        let last_col_idx = self.chart.columns.len() - 1;
        for state in &self.chart.columns[last_col_idx].states {
            if self.is_complete(state) {
                let rule = &self.grammar.rules[state.rule_id.0];
                if rule.lhs == self.start_symbol && state.s_col == ColumnID(0) {
                    return true;
                }
            }
        }
        false
    }

    #[inline]
    fn add_state(&mut self, col_idx: usize, state: State) -> bool {
        self.chart.columns[col_idx].add_state(state)
    }

    // =========================================================================
    // Core Earley Loop
    // =========================================================================

    fn fill_chart(&mut self) {
        let mut i = 0;
        while i < self.chart.columns.len() {
            let mut j = 0;
            // Note: self.chart.columns[i].states grows during iteration
            while j < self.chart.columns[i].states.len() {
                let state = self.chart.columns[i].states[j];

                if self.is_complete(&state) {
                    self.leo_complete(i, state);
                } else {
                    let next_sym = self.next_symbol(&state);
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

    // =========================================================================
    // Standard Operations
    // =========================================================================

    fn predict(&mut self, col_idx: usize, sym: NumSymbol, state: &State) {
        let rule_ids = if let Some(ids) = self.grammar.lookup.get(&sym) {
            ids.clone()
        } else {
            Vec::new()
        };

        for rule_id in rule_ids {
            let new_state = State::new(rule_id, 0, ColumnID(col_idx));
            self.add_state(col_idx, new_state);
        }

        if self.nullables.contains(&sym) {
            let advanced_state = self.advance(state);
            self.add_state(col_idx, advanced_state);
        }
    }

    fn scan(&mut self, col_idx: usize, state: &State, token: NumSymbol) {
        if self.chart.columns[col_idx].token == Some(token) {
            let advanced_state = self.advance(state);
            self.add_state(col_idx, advanced_state);
        }
    }

    fn earley_complete(&mut self, col_idx: usize, state: State) {
        let state_name = self.grammar.rules[state.rule_id.0].lhs;
        let s_col_idx = state.s_col.0;

        // Find parents in start column waiting for this NonTerminal
        let mut parents = Vec::new();
        for st in &self.chart.columns[s_col_idx].states {
            if let Some(next) = self.next_symbol(st) {
                if *next == state_name {
                    parents.push(*st);
                }
            }
        }

        for parent in parents {
            let advanced = self.advance(&parent);
            self.add_state(col_idx, advanced);
        }
    }

    // =========================================================================
    // Leo Optimizations
    // =========================================================================

    fn leo_complete(&mut self, col_idx: usize, state: State) {
        if let Some(top_state) = self.deterministic_reduction(state, col_idx) {
            self.add_state(col_idx, top_state);
        } else {
            self.earley_complete(col_idx, state);
        }
    }

    fn deterministic_reduction(&mut self, state: State, col_idx: usize) -> Option<State> {
        self.get_top(state, col_idx)
    }

    /// Recursively finds the top-most item in a deterministic right-recursive chain.
    fn get_top(&mut self, state_a: State, current_col_idx: usize) -> Option<State> {
        let lhs_a = self.grammar.rules[state_a.rule_id.0].lhs;
        let col_s1_idx = state_a.s_col.0;

        // 1. Memoization Check: Did we already find the top for this symbol at that start column?
        // Even on cache hits, record the current child-to-parent link. The same transitive
        // item can be reused at different end columns, and forest reconstruction needs
        // each occurrence to rebuild a tree after Leo skips the intermediate completions.
        if let Some(&cached_top) = self.chart.columns[col_s1_idx].transitives.get(&lhs_a) {
            let _ = self.uniq_postdot(state_a, current_col_idx);
            return Some(cached_top);
        }

        // 2. Find the unique parent (state_b_inc) that satisfies deterministic constraints
        let st_b_inc = self.uniq_postdot(state_a, current_col_idx)?;

        // 3. Advance the parent to create the completed B state
        let st_b = self.advance(&st_b_inc);

        // 4. Pre-cache before recursing to break cycles in grammars with cyclic
        //    unit production chains (e.g., A -> B -> C -> A). Without this, the
        //    recursive call would never hit the cache and loop infinitely.
        self.chart.columns[col_s1_idx].add_transitive(lhs_a, st_b);

        // 5. Recursive Step: Is there a deterministic path above B?
        // If not, B itself is the top.
        let top = self.get_top(st_b, current_col_idx).unwrap_or(st_b);

        // 6. Update cache with actual top
        self.chart.columns[col_s1_idx].add_transitive(lhs_a, top);

        Some(top)
    }

    /// Identifies if `state_a` contributes to a deterministic reduction path.
    /// Returns the UNIQUE parent state (`st_B_inc`) if constraints are met.
    fn uniq_postdot(&mut self, state_a: State, current_col_idx: usize) -> Option<State> {
        let col_s1_idx = state_a.s_col.0;
        let lhs_a = self.grammar.rules[state_a.rule_id.0].lhs;

        // Find unique parent in s_col waiting for lhs_a (avoid Vec allocation)
        let mut unique_parent: Option<State> = None;
        for s in &self.chart.columns[col_s1_idx].states {
            if let Some(next) = self.next_symbol(s) {
                if *next == lhs_a {
                    if unique_parent.is_some() {
                        return None; // More than one parent — not deterministic
                    }
                    unique_parent = Some(*s);
                }
            }
        }

        let parent = unique_parent?;
        // ... match remainder ...

        // Constraint 2: The parent must be "at the end" after consuming A.
        // i.e., dot is at penultimate position.
        let parent_rule_len = self.grammar.rules[parent.rule_id.0].rhs.len();
        if parent.dot != parent_rule_len - 1 {
            // println!("  Rejected by Constraint 2: parent dot {} len {}", parent.dot, parent_rule_len);
            return None;
        }

        // Record that 'parent' was completed by 'state_a' ending at 'current_col_idx'.
        // This is needed later to "expand" the optimized forest.
        let child_link = (state_a, ColumnID(current_col_idx));
        let children = self.postdots.entry(parent).or_default();
        if !children.contains(&child_link) {
            children.push(child_link);
        }

        Some(parent)
    }

    /// Checks whether a specific parent can be reconstructed with `child_sym`
    /// as its final RHS symbol. This is intentionally not a uniqueness check:
    /// SPPF reconstruction may need to replay an exact Leo-skipped parent even
    /// when the column has other parents waiting for the same non-terminal.
    fn can_reconstruct_leo_child(&self, parent: State, child_sym: NumSymbol) -> bool {
        if self.next_symbol(&parent) != Some(&child_sym) {
            return false;
        }

        let parent_rule_len = self.grammar.rules[parent.rule_id.0].rhs.len();
        if parent.dot != parent_rule_len - 1 {
            return false;
        }

        true
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    #[inline]
    fn is_complete(&self, state: &State) -> bool {
        let rule = &self.grammar.rules[state.rule_id.0];
        state.dot >= rule.rhs.len()
    }

    #[inline]
    fn next_symbol(&self, state: &State) -> Option<&NumSymbol> {
        let rule = &self.grammar.rules[state.rule_id.0];
        rule.rhs.get(state.dot)
    }

    #[inline]
    fn advance(&self, state: &State) -> State {
        State {
            rule_id: state.rule_id,
            dot: state.dot + 1,
            s_col: state.s_col,
        }
    }

    // =========================================================================
    // Public API
    // =========================================================================

    pub fn parse(&mut self, input: Vec<u32>) -> Option<ParseTree> {
        let symbols: Vec<NumSymbol> = input.iter().map(|&id| NumSymbol::Terminal(id)).collect();
        if self.recognize_on(symbols) {
            self.extract_one_tree()
        } else {
            None
        }
    }

    pub fn parse_all(&mut self, input: Vec<u32>) -> Vec<ParseTree> {
        let symbols: Vec<NumSymbol> = input.iter().map(|&id| NumSymbol::Terminal(id)).collect();
        if self.recognize_on(symbols) {
            self.extract_all_trees()
        } else {
            Vec::new()
        }
    }

    pub fn extract_one_tree(&mut self) -> Option<ParseTree> {
        let root = self.build_sppf_single()?;
        Some(self.sppf_to_tree_single(&root))
    }

    pub fn extract_all_trees(&mut self) -> Vec<ParseTree> {
        if let Some(root) = self.build_sppf() {
            self.sppf_to_trees_all(&root)
        } else {
            Vec::new()
        }
    }

    /// Build the reverse index lazily, only when needed for forest construction.
    fn build_state_to_cols(&mut self) {
        self.state_to_cols.clear();
        for (col_idx, col) in self.chart.columns.iter().enumerate() {
            for state in &col.states {
                self.state_to_cols.entry(*state).or_default().push(col_idx);
            }
        }
    }

    pub fn build_sppf(&mut self) -> Option<Rc<SPPFNode>> {
        if self.chart.columns.is_empty() {
            return None;
        }

        let last_col = self.chart.columns.len() - 1;
        let success = self.chart.columns[last_col].states.iter().any(|s| {
            self.is_complete(s)
                && self.grammar.rules[s.rule_id.0].lhs == self.start_symbol
                && s.s_col.0 == 0
        });

        if !success {
            return None;
        }

        // Build reverse index lazily before forest construction
        self.build_state_to_cols();

        let mut builder = ForestBuilder::new(self);
        builder.build(self.start_symbol)
    }

    pub fn build_sppf_single(&mut self) -> Option<Rc<SPPFNode>> {
        if self.chart.columns.is_empty() {
            return None;
        }

        let last_col = self.chart.columns.len() - 1;
        let success = self.chart.columns[last_col].states.iter().any(|s| {
            self.is_complete(s)
                && self.grammar.rules[s.rule_id.0].lhs == self.start_symbol
                && s.s_col.0 == 0
        });

        if !success {
            return None;
        }

        self.build_state_to_cols();

        let mut builder = ForestBuilder::new_single(self);
        builder.build(self.start_symbol)
    }

    // --- Tree Extraction Helpers (Unchanged from original logic) ---

    fn sppf_to_tree_single(&self, node: &SPPFNode) -> ParseTree {
        if node.derivations.is_empty() {
            ParseTree::leaf(&node.symbol.to_string().trim_matches('\''))
        } else {
            let children = node.derivations[0]
                .iter()
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
        if nodes.is_empty() {
            return vec![vec![]];
        }
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

    pub fn debug_chart(&self) {
        println!("Chart Debug:");
        for (i, col) in self.chart.columns.iter().enumerate() {
            println!(
                "Column {}: Token: {:?}",
                i,
                col.token
                    .map(|t| self.num_grammar.symbol_to_str(&t).unwrap_or("?"))
            );
            for state in &col.states {
                let rule = &self.grammar.rules[state.rule_id.0];
                let lhs = self.num_grammar.symbol_to_str(&rule.lhs).unwrap_or("?");
                let rhs: Vec<String> = rule
                    .rhs
                    .iter()
                    .map(|s| self.num_grammar.symbol_to_str(s).unwrap_or("?").to_string())
                    .collect();

                // Format: LHS -> . RHS (RuleID) from s_col
                let mut rhs_str = String::new();
                for (k, sym) in rhs.iter().enumerate() {
                    if k == state.dot {
                        rhs_str.push_str("• ");
                    }
                    rhs_str.push_str(sym);
                    rhs_str.push(' ');
                }
                if state.dot == rhs.len() {
                    rhs_str.push_str("•");
                }

                println!(
                    "  {} -> {} ({:?}) from {}",
                    lhs, rhs_str, state.rule_id, state.s_col.0
                );
            }
        }
    }
}

// ============================================================================
// SPPF Nodes & Builder
// ============================================================================

#[derive(Debug)]
pub struct SPPFNode {
    pub symbol: ParseSymbol,
    pub start_idx: usize,
    pub end_idx: usize,
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

struct ForestBuilder<'a> {
    parser: &'a LeoParser,
    memo: HashMap<(u32, bool, usize, usize), Rc<SPPFNode>>,
    in_progress: HashSet<(u32, bool, usize, usize)>,
    attempt_count: HashMap<(u32, bool, usize, usize), u32>,
    single: bool,
}

impl<'a> ForestBuilder<'a> {
    fn new(parser: &'a LeoParser) -> Self {
        ForestBuilder {
            parser,
            memo: HashMap::default(),
            in_progress: HashSet::default(),
            attempt_count: HashMap::default(),
            single: false,
        }
    }

    fn new_single(parser: &'a LeoParser) -> Self {
        ForestBuilder {
            parser,
            memo: HashMap::default(),
            in_progress: HashSet::default(),
            attempt_count: HashMap::default(),
            single: true,
        }
    }

    fn build(&mut self, root_symbol: NumSymbol) -> Option<Rc<SPPFNode>> {
        let end = self.parser.chart.columns.len().saturating_sub(1);
        self.find_node(root_symbol, 0, end, None)
    }

    fn find_node(
        &mut self,
        sym: NumSymbol,
        start: usize,
        end: usize,
        hint_state: Option<State>,
    ) -> Option<Rc<SPPFNode>> {
        let key = (sym.id(), sym.is_terminal(), start, end);

        if let Some(node) = self.memo.get(&key) {
            return Some(node.clone());
        }

        // Limit retries per key to prevent infinite loops from repeated
        // failed find_node calls in ambiguous grammars
        let count = self.attempt_count.entry(key).or_insert(0);
        *count += 1;
        if *count > 20 {
            return None;
        }

        if self.in_progress.contains(&key) {
            return None;
        }
        self.in_progress.insert(key);

        let parse_sym = match sym {
            NumSymbol::Terminal(id) => {
                ParseSymbol::Terminal(self.parser.num_grammar.terminal_str(id)?.to_string())
            }
            NumSymbol::NonTerminal(id) => {
                ParseSymbol::NonTerminal(self.parser.num_grammar.non_terminal_str(id)?.to_string())
            }
        };

        let mut node = SPPFNode::new(parse_sym, start, end);
        let mut added = false;

        // 1. Hint Processing (Virtual States)
        if let Some(state) = hint_state {
            let rule = &self.parser.grammar.rules[state.rule_id.0];
            // Ensure hint matches request
            if rule.lhs == sym && state.s_col.0 == start {
                let paths = self.walk_back(state.rule_id, rule.rhs.len(), end, start);
                for children in paths {
                    node.add_derivation(children);
                    added = true;
                    if self.single {
                        break;
                    }
                }
            }
        }

        // 2. Chart Processing (Existing States)
        if !(self.single && added) {
            match sym {
                NumSymbol::Terminal(_) => {
                    if start + 1 == end && self.parser.input.get(start) == Some(&sym) {
                        node.add_derivation(vec![]);
                        added = true;
                    }
                }
                NumSymbol::NonTerminal(_) => {
                    let col = &self.parser.chart.columns[end];
                    for state in &col.states {
                        // Check for completion of this symbol starting at 'start'
                        if state.s_col.0 == start {
                            let rule = &self.parser.grammar.rules[state.rule_id.0];
                            if rule.lhs == sym && state.dot >= rule.rhs.len() {
                                let paths =
                                    self.walk_back(state.rule_id, rule.rhs.len(), end, start);
                                for children in paths {
                                    node.add_derivation(children);
                                    added = true;
                                    if self.single {
                                        break;
                                    }
                                }
                                if self.single && added {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.in_progress.remove(&key);

        if added {
            let rc = Rc::new(node);
            self.memo.insert(key, rc.clone());
            Some(rc)
        } else {
            None
        }
    }

    fn walk_back(
        &mut self,
        rule_id: RuleID,
        dot: usize,
        current_end: usize,
        target_start: usize,
    ) -> Vec<Vec<Rc<SPPFNode>>> {
        if dot == 0 {
            return if current_end == target_start {
                vec![vec![]]
            } else {
                vec![]
            };
        }

        let rule = &self.parser.grammar.rules[rule_id.0];
        let child_sym = rule.rhs[dot - 1];
        let mut results = Vec::new();

        // Standard Chart Candidates
        // Bug 2 fix: use HashSet to deduplicate split points — multiple completion rules for the
        // same symbol can produce the same s_col value, which would otherwise cause the same
        // derivation to be processed (and counted) multiple times.
        let mut split_candidates: HashSet<usize> = HashSet::default();
        match child_sym {
            NumSymbol::Terminal(_) => {
                let k = current_end.saturating_sub(1);
                split_candidates.insert(k);
            }
            NumSymbol::NonTerminal(_) => {
                for st in &self.parser.chart.columns[current_end].states {
                    let st_rule = &self.parser.grammar.rules[st.rule_id.0];
                    if st_rule.lhs == child_sym && st.dot >= st_rule.rhs.len() {
                        split_candidates.insert(st.s_col.0);
                    }
                }
            }
        };

        // Virtual Candidates (from Leo Optimization)
        // We look for children that reduced to this specific parent state via postdots.
        let pred_state = State {
            rule_id,
            dot: dot - 1,
            s_col: ColumnID(target_start),
        };
        let mut virtual_matches = Vec::new();

        if let Some(virtual_children) = self.parser.postdots.get(&pred_state) {
            for (child_state, child_end_col) in virtual_children {
                if child_end_col.0 == current_end {
                    virtual_matches.push((child_state.s_col.0, *child_state));
                }
            }
        }

        // Process Standard
        // Track which split points were successfully processed so the virtual path
        // doesn't re-process the same k and produce duplicate derivations (Bug 1 fix).
        let mut processed_k: HashSet<usize> = HashSet::default();
        for k in &split_candidates {
            if *k < target_start {
                continue;
            }
            let prev_st = State {
                rule_id,
                dot: dot - 1,
                s_col: ColumnID(target_start),
            };
            if self.parser.chart.columns[*k].lookup.contains_key(&prev_st) {
                if let Some(child_node) = self.find_node(child_sym, *k, current_end, None) {
                    let prefix_paths = self.walk_back(rule_id, dot - 1, *k, target_start);
                    for mut path in prefix_paths {
                        path.push(child_node.clone());
                        results.push(path);
                        if self.single {
                            return results;
                        }
                    }
                    processed_k.insert(*k);
                }
            }
        }

        // Process Virtuals (Lazy Expansion from Postdots)
        // Bug 1 fix: skip k values already successfully handled by the standard path above.
        // Leo provenance can describe the same span that already exists in the chart, so both
        // paths would find the same derivation.
        for (k, child_state) in virtual_matches {
            if k < target_start {
                continue;
            }
            if processed_k.contains(&k) {
                continue;
            }
            if let Some(child_node) = self.find_node(child_sym, k, current_end, Some(child_state)) {
                let prefix_paths = self.walk_back(rule_id, dot - 1, k, target_start);
                for mut path in prefix_paths {
                    path.push(child_node.clone());
                    results.push(path);
                    if self.single {
                        return results;
                    }
                }
            }
        }

        // 3. Implicit Forest / Leo Reconstruction
        // If we found no matches in the chart or postdots, we might be inside a Leo optimized chain
        // where the intermediate link was not recorded (due to Memoization).
        // We attempt to infer the missing link.
        // We are looking for a state `C` (instance of `child_sym`) spanning `k..current_end`
        // such that `C` deterministically reduces to `pred_state` (Parent).
        // Since `pred_state` exists at `k`, we iterate `k` (split_candidates).
        if results.is_empty() {
            if let NumSymbol::NonTerminal(_) = child_sym {
                if let Some(child_rules) = self.parser.grammar.lookup.get(&child_sym) {
                    // Implicit Reconstruction: Scan potentially skipped split points
                    let expected_parent = State {
                        rule_id,
                        dot: dot - 1,
                        s_col: ColumnID(target_start),
                    };

                    // Optimization: Only scan columns k that actually contain the parent state
                    // This avoids O(N) scan.
                    let cols_empty = Vec::new();
                    let cols = self
                        .parser
                        .state_to_cols
                        .get(&expected_parent)
                        .unwrap_or(&cols_empty);

                    for &k in cols {
                        if k < target_start || k > current_end {
                            continue;
                        }

                        if !self
                            .parser
                            .can_reconstruct_leo_child(expected_parent, child_sym)
                        {
                            continue;
                        }

                        // Try each rule for child_sym. We break after the first
                        // rule that yields a derivation: find_node memoises by
                        // (symbol, start, end), so a later matching rule would
                        // return the same node and duplicate this path.
                        'rule_loop: for &c_rule_id in child_rules {
                            let c_len = self.parser.grammar.rules[c_rule_id.0].rhs.len();
                            let c_state = State {
                                rule_id: c_rule_id,
                                dot: c_len,
                                s_col: ColumnID(k),
                            };

                            if let Some(child_node) =
                                self.find_node(child_sym, k, current_end, Some(c_state))
                            {
                                let prefix_paths =
                                    self.walk_back(rule_id, dot - 1, k, target_start);
                                if !prefix_paths.is_empty() {
                                    for mut path in prefix_paths {
                                        path.push(child_node.clone());
                                        results.push(path);
                                        if self.single {
                                            return results;
                                        }
                                    }
                                    break 'rule_loop;
                                }
                            }
                        }
                    }
                }
            }
        }

        results
    }
}

// ============================================================================
#[test]
fn test_leo_reconstructs_cached_right_recursive_chain() {
    let grammar = load_grammar_from_str(
        r#"{
          "name": "leo_right_recursive",
          "start": "<S>",
          "rules": {
            "<S>": [["<A>"]],
            "<A>": [["a", "<A>"], ["a"]]
          }
        }"#,
    )
    .expect("Failed to load right-recursive grammar");

    let input = grammar.tokenize("aaaa").expect("Failed to tokenize input");
    let mut parser = LeoParser::new(grammar);
    let tree = parser
        .parse(input)
        .expect("Leo should reconstruct a tree for cached reductions");

    assert_eq!(tree.to_flat_string(), "aaaa");
}

#[test]
fn test_leo_parses_gamma2_cached_nullable_chain() {
    let grammar = load_grammar_from_str(
        r#"{
          "name": "gamma2_cached_nullable_chain",
          "start": "<S>",
          "rules": {
            "<S>": [["<B>", "<B>", "<S>", "a"], ["b", "b", "b"]],
            "<B>": [["b", "b", "<B>"], []]
          }
        }"#,
    )
    .expect("Failed to load gamma2 grammar");

    let input = grammar
        .tokenize("bbbbbbbbbbba")
        .expect("Failed to tokenize gamma2 input");
    let mut parser = LeoParser::new(grammar);
    let tree = parser
        .parse(input)
        .expect("Leo should reconstruct a gamma2 tree after cached nullable reductions");

    assert_eq!(tree.to_flat_string(), "bbbbbbbbbbba");
}
