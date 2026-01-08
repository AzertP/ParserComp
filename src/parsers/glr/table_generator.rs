use std::collections::{HashSet, HashMap};
use std::cell::RefCell;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::grammars::{Grammar, NumSymbol};

/// Special symbol representing end of input
pub const END_OF_INPUT: u32 = u32::MAX;

/// Special symbol representing epsilon (empty string)
pub const EPSILON: u32 = u32::MAX - 1;



/// An LR(1) item is a production of the form A -> α·β, with a look-ahead symbol
///
/// Example: `A -> a · b c, d` means:
/// - We have the rule A -> a b c
/// - We've already seen 'a' (dot position = 1)
/// - We expect 'd' to follow after reducing
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Item {
    /// Left-hand side of the production (non-terminal ID)
    pub lhs: u32,
    /// Right-hand side of the production (list of symbols)
    pub rhs: Vec<NumSymbol>,
    /// Position of the dot in the production (0 means at the beginning)
    pub dot: usize,
    /// Look-ahead symbol
    pub look_ahead: NumSymbol,
}

impl Item {
    /// Create a new LR(1) item
    pub fn new(lhs: u32, rhs: Vec<NumSymbol>, dot: usize, look_ahead: NumSymbol) -> Self {
        Item {
            lhs,
            rhs,
            dot,
            look_ahead,
        }
    }

    /// Get the symbol after the dot, or None if dot is at the end
    pub fn next_symbol(&self) -> Option<NumSymbol> {
        if self.dot < self.rhs.len() {
            Some(self.rhs[self.dot])
        } else {
            None
        }
    }

    /// Check if the dot is at the end (item is complete)
    pub fn is_complete(&self) -> bool {
        self.dot >= self.rhs.len()
    }

    /// Create a new item with the dot advanced by one position
    pub fn advance(&self) -> Item {
        Item {
            lhs: self.lhs,
            rhs: self.rhs.clone(),
            dot: self.dot + 1,
            look_ahead: self.look_ahead,
        }
    }
}

impl fmt::Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let before: Vec<String> = self.rhs[..self.dot]
            .iter()
            .map(|s| format!("{:?}", s))
            .collect();
        let after: Vec<String> = self.rhs[self.dot..]
            .iter()
            .map(|s| format!("{:?}", s))
            .collect();
        write!(
            f,
            "NT({}) -> {} · {}, {:?}",
            self.lhs,
            before.join(" "),
            after.join(" "),
            self.look_ahead
        )
    }
}

/// Helper function to convert NumSymbol to a sortable tuple
fn symbol_to_ord(sym: &NumSymbol) -> (u8, u32) {
    match sym {
        NumSymbol::Terminal(id) => (0, *id),
        NumSymbol::NonTerminal(id) => (1, *id),
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare lhs
        match self.lhs.cmp(&other.lhs) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        // Compare rhs element by element
        let self_rhs: Vec<(u8, u32)> = self.rhs.iter().map(symbol_to_ord).collect();
        let other_rhs: Vec<(u8, u32)> = other.rhs.iter().map(symbol_to_ord).collect();
        match self_rhs.cmp(&other_rhs) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        // Compare dot
        match self.dot.cmp(&other.dot) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        // Compare look_ahead
        symbol_to_ord(&self.look_ahead).cmp(&symbol_to_ord(&other.look_ahead))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// State in the LR automaton, consisting of a state ID and a set of items
pub type State = (usize, HashSet<Item>);

/// Cached states and goto map type
type CachedStates = (Vec<State>, HashMap<(usize, NumSymbol), usize>);

/// Special augmented start symbol ID (used internally)
/// We use u32::MAX - 2 to avoid collision with END_OF_INPUT and EPSILON
pub const AUGMENTED_START: u32 = u32::MAX - 2;

/// Generates LR(1) parse tables for GLR parsing
pub struct TableGenerator<'a> {
    grammar: &'a Grammar,
    augmented_start: u32,
    original_start: u32,
    nullable: HashSet<u32>,
    first: HashMap<u32, HashSet<NumSymbol>>,

    /// Cached states and goto map (computed lazily)
    cached_states: RefCell<Option<CachedStates>>,
    /// SPPF for handling epsilon derivations and right-nulled items
    sppf: SPPF,
}

impl<'a> TableGenerator<'a> {
    /// Create a new TableGenerator for the given grammar
    pub fn new(grammar: &'a Grammar) -> Self {
        // Compute nullable set
        let nullable = Self::compute_nullable(grammar);

        // Compute FIRST sets
        let first = Self::compute_first(grammar, &nullable);

        // Build SPPF for epsilon handling
        let sppf = SPPF::new(grammar, &nullable);

        TableGenerator {
            grammar,
            augmented_start: AUGMENTED_START,
            original_start: grammar.start,
            nullable,
            first,
            cached_states: RefCell::new(None),
            sppf,
        }
    }

    /// Compute the set of nullable non-terminals
    ///
    /// A non-terminal is nullable if it can derive the empty string (epsilon)
    pub fn compute_nullable(grammar: &Grammar) -> HashSet<u32> {
        let mut nullable: HashSet<u32> = HashSet::default();
        let mut changed = true;

        while changed {
            changed = false;
            for (&nt, productions) in &grammar.rules {
                if nullable.contains(&nt) {
                    continue;
                }

                for production in productions {
                    // Empty production -> nullable
                    if production.is_empty() {
                        nullable.insert(nt);
                        changed = true;
                        break;
                    }

                    // All symbols in production are nullable non-terminals -> nullable
                    let all_nullable = production.iter().all(|sym| match sym {
                        NumSymbol::NonTerminal(id) => nullable.contains(id),
                        NumSymbol::Terminal(_) => false,
                    });

                    if all_nullable {
                        nullable.insert(nt);
                        changed = true;
                        break;
                    }
                }
            }
        }

        nullable
    }

    /// Compute FIRST sets for all non-terminals
    /// FIRST(X) is the set of terminals that can begin strings derived from X
    fn compute_first(
        grammar: &Grammar,
        nullable: &HashSet<u32>,
    ) -> HashMap<u32, HashSet<NumSymbol>> {
        let mut first: HashMap<u32, HashSet<NumSymbol>> = HashMap::default();

        // Initialize FIRST sets
        for &nt in grammar.rules.keys() {
            let mut set = HashSet::default();
            if nullable.contains(&nt) {
                set.insert(NumSymbol::Terminal(EPSILON));
            }
            first.insert(nt, set);
        }

        let mut changed = true;
        while changed {
            changed = false;

            for (&lhs, productions) in &grammar.rules {
                for production in productions {
                    // Skip empty productions (already handled)
                    if production.is_empty() {
                        continue;
                    }

                    for &symbol in production {
                        match symbol {
                            NumSymbol::NonTerminal(nt_id) => {
                                // Add FIRST(nt_id) - {epsilon} to FIRST(lhs)
                                let nt_first = first.get(&nt_id).cloned().unwrap_or_default();
                                for sym in &nt_first {
                                    if *sym != NumSymbol::Terminal(EPSILON) {
                                        if first.get_mut(&lhs).unwrap().insert(*sym) {
                                            changed = true;
                                        }
                                    }
                                }

                                // If nt is not nullable, stop
                                if !nullable.contains(&nt_id) {
                                    break;
                                }
                            }
                            NumSymbol::Terminal(t_id) => {
                                // Add terminal to FIRST(lhs)
                                if first
                                    .get_mut(&lhs)
                                    .unwrap()
                                    .insert(NumSymbol::Terminal(t_id))
                                {
                                    changed = true;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        first
    }

    /// Calculate FIRST for a sequence of symbols
    fn calculate_first_for_sequence(&self, symbols: &[NumSymbol]) -> HashSet<NumSymbol> {
        if symbols.is_empty() {
            let mut result = HashSet::default();
            result.insert(NumSymbol::Terminal(EPSILON));
            return result;
        }

        let mut result = HashSet::default();
        let mut all_nullable = true;

        for &symbol in symbols {
            match symbol {
                NumSymbol::NonTerminal(nt_id) => {
                    if let Some(nt_first) = self.first.get(&nt_id) {
                        for &sym in nt_first {
                            if sym != NumSymbol::Terminal(EPSILON) {
                                result.insert(sym);
                            }
                        }
                    }

                    if !self.nullable.contains(&nt_id) {
                        all_nullable = false;
                        break;
                    }
                }
                NumSymbol::Terminal(t_id) => {
                    result.insert(NumSymbol::Terminal(t_id));
                    all_nullable = false;
                    break;
                }
            }
        }

        if all_nullable {
            result.insert(NumSymbol::Terminal(EPSILON));
        }

        result
    }

    /// Compute the closure of a set of items
    /// For each item A -> α·Bβ, a in the set, add all items B -> ·γ, b
    /// where b is in FIRST(βa)
    fn find_closure(&self, items: &[Item]) -> HashSet<Item> {
        let mut result: HashSet<Item> = items.iter().cloned().collect();
        let mut changed = true;

        while changed {
            changed = false;
            let current_items: Vec<Item> = result.iter().cloned().collect();

            for item in current_items {
                // Skip if dot is at the end
                if item.is_complete() {
                    continue;
                }

                // Get the symbol after the dot
                if let Some(NumSymbol::NonTerminal(next_nt)) = item.next_symbol() {
                    // Calculate FIRST(βa) where β is the rest after the dot, a is look-ahead
                    let mut beta_a: Vec<NumSymbol> = item.rhs[item.dot + 1..].to_vec();
                    beta_a.push(item.look_ahead);

                    let first_set = self.calculate_first_for_sequence(&beta_a);

                    // Add items for each production of the next non-terminal
                    if let Some(productions) = self.grammar.rules.get(&next_nt) {
                        for production in productions {
                            for &look_ahead in &first_set {
                                // Skip epsilon as look-ahead
                                if look_ahead == NumSymbol::Terminal(EPSILON) {
                                    continue;
                                }

                                let new_item =
                                    Item::new(next_nt, production.clone(), 0, look_ahead);
                                if result.insert(new_item) {
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Compute the transition from a state with a given symbol
    fn transition(&self, state: &HashSet<Item>, symbol: NumSymbol) -> HashSet<Item> {
        let mut items = Vec::new();

        for item in state {
            if let Some(next_sym) = item.next_symbol() {
                if next_sym == symbol {
                    items.push(item.advance());
                }
            }
        }

        self.find_closure(&items)
    }

    /// Generate all LR(1) automaton states
    ///
    /// Returns:
    /// - A list of states (id, item set)
    /// - A GOTO map: (state_id, symbol) -> next_state_id
    ///
    /// Results are cached after first computation.
    pub fn generate_states(&self) -> (Vec<State>, HashMap<(usize, NumSymbol), usize>) {
        // Check if we have a cached result
        if let Some(cached) = self.cached_states.borrow().as_ref() {
            return cached.clone();
        }

        // Compute the states
        let result = self.compute_states();

        // Cache the result
        *self.cached_states.borrow_mut() = Some(result.clone());

        result
    }

    /// Internal method to compute states (called once and cached)
    fn compute_states(&self) -> (Vec<State>, HashMap<(usize, NumSymbol), usize>) {
        // Initial state: [S' -> ·S, $] and its closure
        // S' is the augmented start symbol, S is the original start symbol
        // The augmented production is: S' -> S
        let augmented_production = vec![NumSymbol::NonTerminal(self.original_start)];

        let initial_item = Item::new(
            self.augmented_start,
            augmented_production,
            0,
            NumSymbol::Terminal(END_OF_INPUT),
        );

        let initial_state = (0, self.find_closure(&[initial_item]));

        let mut states: Vec<State> = vec![initial_state.clone()];
        let mut unprocessed: Vec<State> = vec![initial_state];
        let mut goto_map: HashMap<(usize, NumSymbol), usize> = HashMap::default();

        while let Some((state_id, state_items)) = unprocessed.pop() {
            // Collect all symbols that appear after dots in this state
            // Sort them for deterministic state generation (matching Python's sorted())
            let mut next_symbols: Vec<NumSymbol> = state_items
                .iter()
                .filter_map(|item| item.next_symbol())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            next_symbols.sort_by(|a, b| symbol_to_ord(a).cmp(&symbol_to_ord(b)));

            for symbol in next_symbols {
                let next_state_items = self.transition(&state_items, symbol);

                if next_state_items.is_empty() {
                    continue;
                }

                // Check if this state already exists
                let existing_idx = states
                    .iter()
                    .position(|(_, items)| *items == next_state_items);

                match existing_idx {
                    Some(idx) => {
                        goto_map.insert((state_id, symbol), idx);
                    }
                    None => {
                        let new_state_id = states.len();
                        let new_state = (new_state_id, next_state_items);
                        states.push(new_state.clone());
                        unprocessed.push(new_state);
                        goto_map.insert((state_id, symbol), new_state_id);
                    }
                }
            }
        }

        (states, goto_map)
    }

    /// Check if a sequence of symbols is nullable
    fn is_sequence_nullable(&self, symbols: &[NumSymbol]) -> bool {
        symbols.iter().all(|sym| match sym {
            NumSymbol::NonTerminal(nt_id) => self.nullable.contains(nt_id),
            NumSymbol::Terminal(_) => false,
        })
    }

    /// Generate the LR(1) parse table
    ///
    /// Returns a table where:
    /// - table[state_id][symbol] = list of possible actions
    pub fn generate_parse_table(&self) -> HashMap<usize, HashMap<NumSymbol, Vec<Action>>> {
        let (states, goto_map) = self.generate_states();

        let mut table: HashMap<usize, HashMap<NumSymbol, Vec<Action>>> = HashMap::default();

        // Initialize table
        for (state_id, _) in &states {
            table.insert(*state_id, HashMap::default());
        }

        // Add shift and goto actions
        for ((state_id, symbol), next_state) in &goto_map {
            table
                .get_mut(state_id)
                .unwrap()
                .entry(*symbol)
                .or_insert_with(Vec::new)
                .push(Action::Shift(*next_state));
        }

        // Add reduce actions
        for (state_id, state_items) in &states {
            for item in state_items {
                // Dot at the end -> reduce action
                if item.is_complete() {
                    if item.lhs == self.augmented_start {
                        // Accept action for augmented start symbol (S' -> S·)
                        table
                            .get_mut(state_id)
                            .unwrap()
                            .entry(NumSymbol::Terminal(END_OF_INPUT))
                            .or_insert_with(Vec::new)
                            .push(Action::Accept);
                    } else {
                        // Reduce action with SPPF label
                        // If dot == 0 (epsilon production), use SPPF.I[lhs]
                        // Otherwise use 0
                        let sppf_label = if item.dot == 0 {
                            self.sppf.get_single(item.lhs)
                        } else {
                            0
                        };
                        let action = Action::Reduce(item.lhs, item.dot, sppf_label);
                        table
                            .get_mut(state_id)
                            .unwrap()
                            .entry(item.look_ahead)
                            .or_insert_with(Vec::new)
                            .push(action);
                    }
                } else {
                    // Right-nulled items: if remainder after dot is nullable
                    let right_seq = &item.rhs[item.dot..];
                    if self.is_sequence_nullable(right_seq) {
                        if item.lhs == self.augmented_start {
                            table
                                .get_mut(state_id)
                                .unwrap()
                                .entry(item.look_ahead)
                                .or_insert_with(Vec::new)
                                .push(Action::Accept);
                        } else {
                            // Get the SPPF label for the nullable sequence
                            let nt_ids: Vec<u32> = right_seq
                                .iter()
                                .filter_map(|sym| match sym {
                                    NumSymbol::NonTerminal(id) => Some(*id),
                                    _ => None,
                                })
                                .collect();
                            let sppf_label = self.sppf.get_sequence(&nt_ids);
                            let action = Action::Reduce(item.lhs, item.dot, sppf_label);
                            table
                                .get_mut(state_id)
                                .unwrap()
                                .entry(item.look_ahead)
                                .or_insert_with(Vec::new)
                                .push(action);
                        }
                    }
                }
            }
        }

        table
    }

    /// Get the number of states in the automaton
    pub fn state_count(&self) -> usize {
        self.generate_states().0.len()
    }

    /// Check if the grammar has any conflicts
    pub fn has_conflicts(&self) -> bool {
        let table = self.generate_parse_table();
        for actions in table.values() {
            for action_list in actions.values() {
                if action_list.len() > 1 {
                    return true;
                }
            }
        }
        false
    }

    /// Export the parse table to a CSV file
    ///
    /// Format:
    /// - Header row: "state", followed by all symbols
    /// - Each subsequent row: state_id, followed by actions for each symbol
    /// - Multiple actions for the same cell are separated by "/"
    ///
    /// # Arguments
    /// * `path` - Path to the output CSV file
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(io::Error)` on file I/O failure
    pub fn export_to_csv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let table = self.generate_parse_table();

        // Collect and sort state IDs
        let mut state_ids: Vec<usize> = table.keys().copied().collect();
        state_ids.sort();

        // Collect all symbols from the table
        let mut all_symbols: HashSet<NumSymbol> = HashSet::default();
        for actions in table.values() {
            for &symbol in actions.keys() {
                all_symbols.insert(symbol);
            }
        }

        // Sort symbols for consistent column ordering
        let mut symbols: Vec<NumSymbol> = all_symbols.into_iter().collect();
        symbols.sort_by(|a, b| symbol_to_ord(a).cmp(&symbol_to_ord(b)));

        // Create file and buffered writer
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header row
        write!(writer, "state")?;
        for symbol in &symbols {
            write!(writer, ",{}", self.format_symbol(symbol))?;
        }
        writeln!(writer)?;

        // Write data rows
        for state_id in state_ids {
            write!(writer, "{}", state_id)?;

            if let Some(state_actions) = table.get(&state_id) {
                for symbol in &symbols {
                    write!(writer, ",")?;
                    if let Some(actions) = state_actions.get(symbol) {
                        let action_strs: Vec<String> =
                            actions.iter().map(|a| self.format_action(a)).collect();
                        write!(writer, "{}", action_strs.join("/"))?;
                    }
                }
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Export the parse table to a CSV file in numeric format
    ///
    /// This format matches GLR_num.py:
    /// - Symbols are represented as integers
    /// - Actions use dot-separated format: "p.{state}", "r.{lhs}.{dot}.{label}", "acc"
    pub fn export_to_csv_numeric<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let table = self.generate_parse_table();

        // Collect and sort state IDs
        let mut state_ids: Vec<usize> = table.keys().copied().collect();
        state_ids.sort();

        // Collect all symbols from the table
        let mut all_symbols: HashSet<NumSymbol> = HashSet::default();
        for actions in table.values() {
            for &symbol in actions.keys() {
                all_symbols.insert(symbol);
            }
        }

        // Sort symbols for consistent column ordering
        let mut symbols: Vec<NumSymbol> = all_symbols.into_iter().collect();
        symbols.sort_by(|a, b| symbol_to_ord(a).cmp(&symbol_to_ord(b)));

        // Create file and buffered writer
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header row with numeric symbols
        write!(writer, "state")?;
        for symbol in &symbols {
            write!(writer, ",{}", self.format_symbol_numeric(symbol))?;
        }
        writeln!(writer)?;

        // Write data rows
        for state_id in state_ids {
            write!(writer, "{}", state_id)?;

            if let Some(state_actions) = table.get(&state_id) {
                for symbol in &symbols {
                    write!(writer, ",")?;
                    if let Some(actions) = state_actions.get(symbol) {
                        let action_strs: Vec<String> =
                            actions.iter().map(|a| self.format_action(a)).collect();
                        write!(writer, "{}", action_strs.join("/"))?;
                    }
                }
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Format a symbol for CSV output
    fn format_symbol(&self, symbol: &NumSymbol) -> String {
        match symbol {
            NumSymbol::Terminal(id) if *id == END_OF_INPUT => "$".to_string(),
            NumSymbol::Terminal(id) if *id == EPSILON => "ε".to_string(),
            NumSymbol::Terminal(id) => self
                .grammar
                .terminals
                .get_str(*id)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("t{}", id)),
            NumSymbol::NonTerminal(id) => self
                .grammar
                .non_terminals
                .get_str(*id)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("<{}>", id)),
        }
    }

    /// Format an action for CSV output
    ///
    /// - Shift: "p.{state}"  
    /// - Reduce: "r.{lhs_id}.{dot}.{sppf_label}" (lhs is -(id+1) for NTs to avoid -0)
    /// - Accept: "acc"
    fn format_action(&self, action: &Action) -> String {
        match action {
            Action::Shift(state) => format!("p.{}", state),
            Action::Reduce(lhs, dot, label) => {
                // Use -(lhs + 1) to avoid -0 issue, matching format_symbol_numeric
                format!("r.{}.{}.{}", -((*lhs as i64) + 1), dot, label)
            }
            Action::Accept => "acc".to_string(),
        }
    }

    /// Format a symbol for CSV output (numeric format)
    ///
    /// - End of input: 0
    /// - Terminals: positive integers starting from 1 (ID + 1)
    /// - Non-terminals: negative integers (-ID - 1 to avoid -0)
    fn format_symbol_numeric(&self, symbol: &NumSymbol) -> String {
        match symbol {
            NumSymbol::Terminal(id) if *id == END_OF_INPUT => "0".to_string(),
            NumSymbol::Terminal(id) => format!("{}", *id + 1), // Offset by 1
            NumSymbol::NonTerminal(id) => format!("{}", -((*id as i64) + 1)), // -1, -2, -3, ...
        }
    }
}

/// Key for nullable sequence labels - can be a single NT or a tuple of NTs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NullableLabel {
    /// Single nullable non-terminal
    Single(u32),
    /// Sequence of nullable non-terminals
    Sequence(Vec<u32>),
    /// Empty sequence (epsilon)
    Epsilon,
}

/// Represents an action in the parse table
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Shift: push state k onto the stack
    Shift(usize),
    /// Reduce: reduce using rule A -> α with dot position and SPPF label
    /// Format: Reduce(lhs_nt_id, dot_position, sppf_label)
    Reduce(u32, usize, usize),
    /// Accept: parsing complete
    Accept,
}

/// SPPF structure for handling epsilon derivations
///
/// This is equivalent to the Python SPPF class in GLR_num.py
/// It builds an epsilon-SPPF and provides the I mapping function
pub struct SPPF {
    /// Counter for node IDs
    #[allow(dead_code)]
    counter: usize,
    /// The I function: maps nullable labels to node IDs
    /// Used in reduce actions for right-nulled items
    pub i_map: HashMap<NullableLabel, usize>,
}

impl SPPF {
    /// Create a new SPPF for the given grammar
    pub fn new(grammar: &Grammar, nullable: &HashSet<u32>) -> Self {
        let (counter, i_map) = Self::build_epsilon_sppf(grammar, nullable);
        SPPF { counter, i_map }
    }

    fn build_epsilon_sppf(
        grammar: &Grammar,
        nullable: &HashSet<u32>,
    ) -> (usize, HashMap<NullableLabel, usize>) {
        let mut i_map: HashMap<NullableLabel, usize> = HashMap::default();
        let mut counter: usize = 1;

        // Step 1: Add all nullable non-terminals
        let mut sorted_nullable: Vec<u32> = nullable.iter().copied().collect();
        sorted_nullable.sort_unstable(); // efficient sort for primitives

        for nt in sorted_nullable {
            i_map.insert(NullableLabel::Single(nt), counter);
            counter += 1;
        }

        // Step 2: Add partial nullable sequences
        let mut sorted_lhs: Vec<u32> = grammar.rules.keys().copied().collect();
        sorted_lhs.sort_unstable();

        for lhs in sorted_lhs {
            let productions = grammar.rules.get(&lhs).unwrap(); 

            for rhs in productions {
                for i in 1..rhs.len() {
                    let partial_rhs = &rhs[i..];

                    let all_nullable = partial_rhs.iter().all(|sym| match sym {
                        NumSymbol::NonTerminal(nt_id) => nullable.contains(nt_id),
                        NumSymbol::Terminal(_) => false,
                    });

                    if all_nullable {
                        let nt_ids: Vec<u32> = partial_rhs
                            .iter()
                            .filter_map(|sym| match sym {
                                NumSymbol::NonTerminal(id) => Some(*id),
                                _ => None,
                            })
                            .collect();

                        if nt_ids.len() == 1 { 
                            // Add single child
                            i_map.insert(NullableLabel::Single(nt_ids[0]), counter);
                            continue; 
                        }

                        let label = NullableLabel::Sequence(nt_ids);
                        // Check logic remains the same
                        if !i_map.contains_key(&label) {
                            i_map.insert(label, counter);
                            counter += 1;
                        }
                    }
                }
            }
        }

        (counter, i_map)
    }

    /// Get the SPPF node ID for a nullable label
    /// Returns 0 if not found (epsilon node)
    pub fn get_label(&self, label: &NullableLabel) -> usize {
        *self.i_map.get(label).unwrap_or(&0)
    }

    /// Get the SPPF node ID for a single nullable non-terminal
    pub fn get_single(&self, nt: u32) -> usize {
        self.get_label(&NullableLabel::Single(nt))
    }

    /// Get the SPPF node ID for a sequence of nullable non-terminals
    pub fn get_sequence(&self, nts: &[u32]) -> usize {
        if nts.len() == 1 {
            self.get_single(nts[0])
        } else {
            self.get_label(&NullableLabel::Sequence(nts.to_vec()))
        }
    }
}