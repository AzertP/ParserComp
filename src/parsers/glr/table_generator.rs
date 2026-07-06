use std::cell::RefCell;
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::grammars::{Grammar, NumSymbol};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Special symbol representing end of input
pub const END_OF_INPUT: u32 = u32::MAX;

/// Special symbol representing epsilon (empty string)
pub const EPSILON: u32 = u32::MAX - 1;

const MAX_CACHED_CLOSURE_ITEMS: usize = 256;
const MAX_CACHED_CLOSURES: usize = 20_000;

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
    pub rhs: Arc<[NumSymbol]>,
    /// Position of the dot in the production (0 means at the beginning)
    pub dot: usize,
    /// Look-ahead symbol
    pub look_ahead: NumSymbol,
}

impl Item {
    /// Create a new LR(1) item
    pub fn new(lhs: u32, rhs: Vec<NumSymbol>, dot: usize, look_ahead: NumSymbol) -> Self {
        Self::from_shared_rhs(lhs, Arc::from(rhs.into_boxed_slice()), dot, look_ahead)
    }

    fn from_shared_rhs(lhs: u32, rhs: Arc<[NumSymbol]>, dot: usize, look_ahead: NumSymbol) -> Self {
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
            rhs: Arc::clone(&self.rhs),
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
        self.lhs
            .cmp(&other.lhs)
            .then_with(|| {
                self.rhs
                    .iter()
                    .map(symbol_to_ord)
                    .cmp(other.rhs.iter().map(symbol_to_ord))
            })
            .then_with(|| self.dot.cmp(&other.dot))
            .then_with(|| symbol_to_ord(&self.look_ahead).cmp(&symbol_to_ord(&other.look_ahead)))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// State in the LR automaton, consisting of a state ID and a set of items
pub type State = (usize, HashSet<Item>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StateKey(Vec<Item>);

impl StateKey {
    fn from_items(items: &HashSet<Item>) -> Self {
        let mut sorted_items: Vec<Item> = items.iter().cloned().collect();
        sorted_items.sort_unstable();
        StateKey(sorted_items)
    }

    fn from_item_slice(items: &[Item]) -> Self {
        let mut sorted_items = items.to_vec();
        sorted_items.sort_unstable();
        StateKey(sorted_items)
    }
}

#[derive(Clone, Copy)]
struct ProgressSnapshot {
    processed: usize,
    states: usize,
    pending: usize,
    gotos: usize,
    first_cache: usize,
    closure_cache: usize,
}

struct ProgressReporter {
    enabled: bool,
    interval: usize,
}

impl ProgressReporter {
    fn from_env() -> Self {
        let enabled = env::var("LR_TABLE_PROGRESS")
            .map(|value| {
                matches!(
                    value.as_str(),
                    "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
                )
            })
            .unwrap_or(false);
        let interval = env::var("LR_TABLE_PROGRESS_INTERVAL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1000);

        Self::new(enabled, interval)
    }

    fn new(enabled: bool, interval: usize) -> Self {
        ProgressReporter {
            enabled,
            interval: interval.max(1),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn should_report(&self, processed: usize) -> bool {
        self.enabled && processed > 0 && processed % self.interval == 0
    }

    fn format_snapshot(&self, snapshot: ProgressSnapshot, elapsed: Duration) -> String {
        format!(
            "lr-table-progress processed={} states={} pending={} gotos={} first_cache={} closure_cache={} elapsed={}s",
            snapshot.processed,
            snapshot.states,
            snapshot.pending,
            snapshot.gotos,
            snapshot.first_cache,
            snapshot.closure_cache,
            elapsed.as_secs()
        )
    }

    fn report(&self, snapshot: ProgressSnapshot, elapsed: Duration) {
        if self.enabled {
            eprintln!("{}", self.format_snapshot(snapshot, elapsed));
        }
    }
}

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
    first_sequence_cache: RefCell<HashMap<Vec<NumSymbol>, HashSet<NumSymbol>>>,
    closure_cache: RefCell<HashMap<StateKey, HashSet<Item>>>,
    production_cache: HashMap<u32, Vec<Arc<[NumSymbol]>>>,

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
        let mut production_cache: HashMap<u32, Vec<Arc<[NumSymbol]>>> = HashMap::default();
        let mut lhs_ids: Vec<u32> = grammar.rules.keys().copied().collect();
        lhs_ids.sort_unstable();

        for lhs in lhs_ids {
            let mut shared_productions = Vec::new();
            if let Some(productions) = grammar.rules.get(&lhs) {
                for production in productions {
                    shared_productions.push(Arc::from(production.clone().into_boxed_slice()));
                }
            }
            production_cache.insert(lhs, shared_productions);
        }

        TableGenerator {
            grammar,
            augmented_start: AUGMENTED_START,
            original_start: grammar.start,
            nullable,
            first,
            first_sequence_cache: RefCell::new(HashMap::default()),
            closure_cache: RefCell::new(HashMap::default()),
            production_cache,
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
        let key = symbols.to_vec();
        if let Some(cached) = self.first_sequence_cache.borrow().get(&key) {
            return cached.clone();
        }

        let result = self.calculate_first_for_sequence_uncached(symbols);
        self.first_sequence_cache
            .borrow_mut()
            .insert(key, result.clone());
        result
    }

    fn calculate_first_for_sequence_uncached(&self, symbols: &[NumSymbol]) -> HashSet<NumSymbol> {
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

    fn calculate_first_for_suffix_and_lookahead(
        &self,
        beta: &[NumSymbol],
        lookahead: NumSymbol,
    ) -> HashSet<NumSymbol> {
        let mut first_set = self.calculate_first_for_sequence(beta);
        if first_set.remove(&NumSymbol::Terminal(EPSILON)) {
            first_set.insert(lookahead);
        }
        first_set
    }

    /// Compute the closure of a set of items
    /// For each item A -> α·Bβ, a in the set, add all items B -> ·γ, b
    /// where b is in FIRST(βa)
    fn find_closure(&self, items: &[Item]) -> HashSet<Item> {
        let kernel_key = StateKey::from_item_slice(items);
        if let Some(cached) = self.closure_cache.borrow().get(&kernel_key) {
            return cached.clone();
        }

        let mut result: HashSet<Item> = items.iter().cloned().collect();
        let mut worklist: Vec<Item> = items.to_vec();

        while let Some(item) = worklist.pop() {
            if item.is_complete() {
                continue;
            }

            if let Some(NumSymbol::NonTerminal(next_nt)) = item.next_symbol() {
                let first_set = self.calculate_first_for_suffix_and_lookahead(
                    &item.rhs[item.dot + 1..],
                    item.look_ahead,
                );

                if let Some(productions) = self.production_cache.get(&next_nt) {
                    for production in productions {
                        for &look_ahead in &first_set {
                            if look_ahead == NumSymbol::Terminal(EPSILON) {
                                continue;
                            }

                            let new_item = Item::from_shared_rhs(
                                next_nt,
                                Arc::clone(production),
                                0,
                                look_ahead,
                            );
                            if result.insert(new_item.clone()) {
                                worklist.push(new_item);
                            }
                        }
                    }
                }
            }
        }

        if result.len() <= MAX_CACHED_CLOSURE_ITEMS {
            let mut closure_cache = self.closure_cache.borrow_mut();
            if closure_cache.len() < MAX_CACHED_CLOSURES {
                closure_cache.insert(kernel_key, result.clone());
            }
        }
        result
    }

    fn transition_kernels(state: &HashSet<Item>) -> Vec<(NumSymbol, Vec<Item>)> {
        let mut grouped: HashMap<NumSymbol, Vec<Item>> = HashMap::default();

        for item in state {
            if let Some(next_sym) = item.next_symbol() {
                grouped.entry(next_sym).or_default().push(item.advance());
            }
        }

        let mut kernels: Vec<(NumSymbol, Vec<Item>)> = grouped.into_iter().collect();
        kernels.sort_by(|(left, _), (right, _)| symbol_to_ord(left).cmp(&symbol_to_ord(right)));
        for (_, items) in &mut kernels {
            items.sort_unstable();
        }
        kernels
    }

    /// Generate all LR(1) automaton states
    ///
    /// Returns:
    /// - A list of states (id, item set)
    /// - A GOTO map: (state_id, symbol) -> next_state_id
    ///
    /// Results are not retained inside the generator. Keeping a second full
    /// automaton copy is too costly for large grammars while exporting tables.
    pub fn generate_states(&self) -> (Vec<State>, HashMap<(usize, NumSymbol), usize>) {
        self.compute_states()
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
        let initial_key = StateKey::from_items(&initial_state.1);

        let mut states: Vec<State> = vec![initial_state.clone()];
        let mut unprocessed: Vec<State> = vec![initial_state];
        let mut state_index: HashMap<StateKey, usize> = HashMap::default();
        state_index.insert(initial_key, 0);
        let mut goto_map: HashMap<(usize, NumSymbol), usize> = HashMap::default();
        let progress = ProgressReporter::from_env();
        let progress_start = Instant::now();
        let mut processed_states = 0usize;

        while let Some((state_id, state_items)) = unprocessed.pop() {
            processed_states += 1;
            for (symbol, kernel_items) in Self::transition_kernels(&state_items) {
                let next_state_items = self.find_closure(&kernel_items);

                if next_state_items.is_empty() {
                    continue;
                }

                let next_state_key = StateKey::from_items(&next_state_items);

                if let Some(&existing_state_id) = state_index.get(&next_state_key) {
                    goto_map.insert((state_id, symbol), existing_state_id);
                } else {
                    let new_state_id = states.len();
                    state_index.insert(next_state_key, new_state_id);
                    let new_state = (new_state_id, next_state_items);
                    states.push(new_state.clone());
                    unprocessed.push(new_state);
                    goto_map.insert((state_id, symbol), new_state_id);
                }
            }

            if progress.should_report(processed_states) {
                progress.report(
                    ProgressSnapshot {
                        processed: processed_states,
                        states: states.len(),
                        pending: unprocessed.len(),
                        gotos: goto_map.len(),
                        first_cache: self.first_sequence_cache.borrow().len(),
                        closure_cache: self.closure_cache.borrow().len(),
                    },
                    progress_start.elapsed(),
                );
            }
        }

        if progress.enabled() {
            progress.report(
                ProgressSnapshot {
                    processed: processed_states,
                    states: states.len(),
                    pending: unprocessed.len(),
                    gotos: goto_map.len(),
                    first_cache: self.first_sequence_cache.borrow().len(),
                    closure_cache: self.closure_cache.borrow().len(),
                },
                progress_start.elapsed(),
            );
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

    /// Generate a standard LR(1) parse table (without Right-Nulled items)
    ///
    /// This is simplified to omit SPPF handling (uses 0 as placeholder)
    pub fn generate_lr1_table(&self) -> HashMap<usize, HashMap<NumSymbol, Vec<Action>>> {
        let (states, goto_map) = self.generate_states();

        let mut table: HashMap<usize, HashMap<NumSymbol, Vec<Action>>> = HashMap::default();

        // Initialize table
        for (state_id, _) in &states {
            table.insert(*state_id, HashMap::default());
        }

        // Add shift and goto actions
        for ((state_id, symbol), next_state) in &goto_map {
            let actions = table
                .get_mut(state_id)
                .unwrap()
                .entry(*symbol)
                .or_insert_with(Vec::new);

            // Check for shift/reduce conflict
            if !actions.is_empty() {
                eprintln!("🚨🚨🚨 CONFLICT DETECTED! 🚨🚨🚨");
                eprintln!(
                    "State {}: Shift/Reduce conflict on symbol '{}'",
                    state_id,
                    self.format_symbol(symbol)
                );
                eprint!("  Existing actions: ");
                for (i, action) in actions.iter().enumerate() {
                    if i > 0 {
                        eprint!(", ");
                    }
                    eprint!("{}", self.format_action(action));
                }
                eprintln!();
                eprintln!("  New action: Shift to state {}", next_state);
                eprintln!("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨");
            }

            actions.push(Action::Shift(*next_state));
        }

        // Add reduce actions
        for (state_id, state_items) in &states {
            for item in state_items {
                // Dot at the end -> reduce action
                if item.is_complete() {
                    if item.lhs == self.augmented_start {
                        // Accept action for augmented start symbol (S' -> S·)
                        let actions = table
                            .get_mut(state_id)
                            .unwrap()
                            .entry(NumSymbol::Terminal(END_OF_INPUT))
                            .or_insert_with(Vec::new);

                        if !actions.is_empty() {
                            eprintln!("🚨🚨🚨 CONFLICT DETECTED! 🚨🚨🚨");
                            eprintln!("State {}: Accept conflict on END_OF_INPUT", state_id);
                            eprint!("  Existing actions: ");
                            for (i, action) in actions.iter().enumerate() {
                                if i > 0 {
                                    eprint!(", ");
                                }
                                eprint!("{}", self.format_action(action));
                            }
                            eprintln!();
                            eprintln!("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨");
                        }

                        actions.push(Action::Accept);
                    } else {
                        // Reduce action with dummy SPPF label (0)
                        let action = Action::Reduce(item.lhs, item.dot, 0);
                        let actions = table
                            .get_mut(state_id)
                            .unwrap()
                            .entry(item.look_ahead)
                            .or_insert_with(Vec::new);

                        // Check for reduce/reduce or shift/reduce conflict
                        if !actions.is_empty() {
                            let conflict_type =
                                if actions.iter().any(|a| matches!(a, Action::Shift(_))) {
                                    "Shift/Reduce"
                                } else {
                                    "Reduce/Reduce"
                                };

                            // Format the reduction for display
                            let lhs_name =
                                self.grammar.non_terminals.get_str(item.lhs).unwrap_or("?");
                            let rhs_strs: Vec<String> =
                                item.rhs.iter().map(|s| self.format_symbol(s)).collect();

                            eprintln!("🚨🚨🚨 CONFLICT DETECTED! 🚨🚨🚨");
                            eprintln!(
                                "State {}: {} conflict on symbol '{}'",
                                state_id,
                                conflict_type,
                                self.format_symbol(&item.look_ahead)
                            );
                            eprint!("  Existing actions: ");
                            for (i, a) in actions.iter().enumerate() {
                                if i > 0 {
                                    eprint!(", ");
                                }
                                eprint!("{}", self.format_action(a));
                            }
                            eprintln!();
                            eprintln!(
                                "  New action: Reduce using rule {} -> {}",
                                lhs_name,
                                rhs_strs.join(" ")
                            );
                            eprintln!("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨");
                        }

                        actions.push(action);
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

    /// Export the LR(1) parse table to a CSV file in numeric format
    pub fn export_lr1_to_csv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let table = self.generate_lr1_table();

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

#[cfg(test)]
mod tests {
    use super::*;

    fn item(lhs: u32, rhs: Vec<NumSymbol>, dot: usize, lookahead: NumSymbol) -> Item {
        Item::new(lhs, rhs, dot, lookahead)
    }

    #[test]
    fn state_key_is_independent_of_hashset_iteration_order() {
        let item_a = item(1, vec![NumSymbol::Terminal(10)], 0, NumSymbol::Terminal(20));
        let item_b = item(
            2,
            vec![NumSymbol::NonTerminal(3)],
            0,
            NumSymbol::Terminal(21),
        );

        let first: HashSet<Item> = [item_a.clone(), item_b.clone()].into_iter().collect();
        let second: HashSet<Item> = [item_b, item_a].into_iter().collect();

        assert_eq!(StateKey::from_items(&first), StateKey::from_items(&second));
    }

    #[test]
    fn state_key_distinguishes_lookahead() {
        let first: HashSet<Item> = [item(
            1,
            vec![NumSymbol::Terminal(10)],
            0,
            NumSymbol::Terminal(20),
        )]
        .into_iter()
        .collect();
        let second: HashSet<Item> = [item(
            1,
            vec![NumSymbol::Terminal(10)],
            0,
            NumSymbol::Terminal(21),
        )]
        .into_iter()
        .collect();

        assert_ne!(StateKey::from_items(&first), StateKey::from_items(&second));
    }

    #[test]
    fn advancing_item_reuses_rhs_storage() {
        let item = item(
            1,
            vec![NumSymbol::Terminal(10), NumSymbol::NonTerminal(20)],
            0,
            NumSymbol::Terminal(30),
        );
        let advanced = item.advance();

        assert!(std::sync::Arc::ptr_eq(&item.rhs, &advanced.rhs));
    }

    #[test]
    fn transition_kernels_group_advanced_items_in_symbol_order() {
        let a_item = item(
            1,
            vec![NumSymbol::Terminal(10), NumSymbol::Terminal(11)],
            0,
            NumSymbol::Terminal(30),
        );
        let b_item = item(
            2,
            vec![NumSymbol::NonTerminal(3), NumSymbol::Terminal(12)],
            0,
            NumSymbol::Terminal(31),
        );
        let another_a_item = item(
            4,
            vec![NumSymbol::Terminal(10), NumSymbol::Terminal(13)],
            0,
            NumSymbol::Terminal(32),
        );
        let state: HashSet<Item> = [a_item, b_item, another_a_item].into_iter().collect();

        let kernels = TableGenerator::transition_kernels(&state);

        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0].0, NumSymbol::Terminal(10));
        assert_eq!(kernels[0].1.len(), 2);
        assert!(kernels[0].1.iter().all(|item| item.dot == 1));
        assert_eq!(kernels[1].0, NumSymbol::NonTerminal(3));
        assert_eq!(kernels[1].1.len(), 1);
        assert_eq!(kernels[1].1[0].dot, 1);
    }

    #[test]
    fn first_sequence_results_are_cached_by_exact_symbol_sequence() {
        let grammar = crate::grammars::load_grammar_from_str(
            r#"{
                "name": "first_cache",
                "start": "<S>",
                "rules": {
                    "<S>": [["<A>", "b"]],
                    "<A>": [["a"], []]
                }
            }"#,
        )
        .expect("load grammar");
        let generator = TableGenerator::new(&grammar);
        let a = grammar.non_terminals.get_id("<A>").expect("A id");
        let b = grammar.terminals.get_id("b").expect("b id");
        let sequence = vec![NumSymbol::NonTerminal(a), NumSymbol::Terminal(b)];

        let first = generator.calculate_first_for_sequence(&sequence);
        let second = generator.calculate_first_for_sequence(&sequence);

        assert_eq!(first, second);
        assert_eq!(generator.first_sequence_cache.borrow().len(), 1);
    }

    #[test]
    fn first_suffix_cache_reuses_nonnullable_beta_across_lookaheads() {
        let grammar = crate::grammars::load_grammar_from_str(
            r#"{
                "name": "first_suffix_cache",
                "start": "<S>",
                "rules": {
                    "<S>": [["<A>", "b"]],
                    "<A>": [["a"]]
                }
            }"#,
        )
        .expect("load grammar");
        let generator = TableGenerator::new(&grammar);
        let b = grammar.terminals.get_id("b").expect("b id");
        let c = NumSymbol::Terminal(100);
        let d = NumSymbol::Terminal(101);
        let beta = [NumSymbol::Terminal(b)];

        let first = generator.calculate_first_for_suffix_and_lookahead(&beta, c);
        let second = generator.calculate_first_for_suffix_and_lookahead(&beta, d);

        assert_eq!(first, second);
        assert_eq!(first, [NumSymbol::Terminal(b)].into_iter().collect());
        assert_eq!(generator.first_sequence_cache.borrow().len(), 1);
    }

    #[test]
    fn first_suffix_uses_lookahead_when_beta_is_nullable() {
        let grammar = crate::grammars::load_grammar_from_str(
            r#"{
                "name": "first_suffix_nullable",
                "start": "<S>",
                "rules": {
                    "<S>": [["<A>"]],
                    "<A>": [["a"], []]
                }
            }"#,
        )
        .expect("load grammar");
        let generator = TableGenerator::new(&grammar);
        let a = grammar.terminals.get_id("a").expect("a id");
        let nt_a = grammar.non_terminals.get_id("<A>").expect("A id");
        let lookahead = NumSymbol::Terminal(100);
        let beta = [NumSymbol::NonTerminal(nt_a)];

        let first = generator.calculate_first_for_suffix_and_lookahead(&beta, lookahead);

        assert_eq!(
            first,
            [NumSymbol::Terminal(a), lookahead].into_iter().collect()
        );
    }

    #[test]
    fn closure_results_are_cached_by_kernel_items() {
        let grammar = crate::grammars::load_grammar_from_str(
            r#"{
                "name": "closure_cache",
                "start": "<S>",
                "rules": {
                    "<S>": [["<A>"]],
                    "<A>": [["a"]]
                }
            }"#,
        )
        .expect("load grammar");
        let generator = TableGenerator::new(&grammar);
        let s = grammar.non_terminals.get_id("<S>").expect("S id");
        let kernel = [Item::new(
            AUGMENTED_START,
            vec![NumSymbol::NonTerminal(s)],
            0,
            NumSymbol::Terminal(END_OF_INPUT),
        )];

        let first = generator.find_closure(&kernel);
        let second = generator.find_closure(&kernel);

        assert_eq!(first, second);
        assert_eq!(generator.closure_cache.borrow().len(), 1);
    }

    #[test]
    fn oversized_closure_results_are_not_cached() {
        let grammar = crate::grammars::simple_grammar();
        let generator = TableGenerator::new(&grammar);
        let kernel: Vec<Item> = (0..=MAX_CACHED_CLOSURE_ITEMS)
            .map(|idx| {
                Item::new(
                    10_000 + idx as u32,
                    vec![NumSymbol::Terminal(idx as u32)],
                    1,
                    NumSymbol::Terminal(END_OF_INPUT),
                )
            })
            .collect();

        let closure = generator.find_closure(&kernel);

        assert_eq!(closure.len(), MAX_CACHED_CLOSURE_ITEMS + 1);
        assert!(generator.closure_cache.borrow().is_empty());
    }

    #[test]
    fn progress_reporter_reports_only_on_interval() {
        let reporter = ProgressReporter::new(true, 4);

        assert!(!reporter.should_report(1));
        assert!(!reporter.should_report(3));
        assert!(reporter.should_report(4));
        assert!(reporter.should_report(8));
    }

    #[test]
    fn progress_reporter_formats_snapshot_counters() {
        let reporter = ProgressReporter::new(true, 1000);
        let snapshot = ProgressSnapshot {
            processed: 2_000,
            states: 3_200,
            pending: 400,
            gotos: 12_345,
            first_cache: 67,
            closure_cache: 89,
        };

        assert_eq!(
            reporter.format_snapshot(snapshot, std::time::Duration::from_secs(125)),
            "lr-table-progress processed=2000 states=3200 pending=400 gotos=12345 first_cache=67 closure_cache=89 elapsed=125s"
        );
    }
}
