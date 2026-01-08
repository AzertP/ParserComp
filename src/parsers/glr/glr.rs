// GLR parser - equivalent to python_files/final_parser/GLR.py

use crate::parsers::glr::table_generator::*;
use crate::grammars::{Grammar, NumSymbol};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;
use std::io;
use std::fs::File;
use std::path::Path;

// use crate::parsers::glr::table_generator::{END_OF_INPUT, EPSILON};

// Use faster hash implementations
type HashMap<K, V> = FxHashMap<K, V>;
type HashSet<T> = FxHashSet<T>;

// ============================================================================
// GSS - Graph Structured Stack
// ============================================================================

/// A node in the Graph-Structured Stack
/// Nodes are identified by their unique ID
#[derive(Clone)]
pub struct GSSNode {
    pub level: usize,
    pub id: usize,
    pub label: usize,
    pub children: Vec<(usize, usize)>,
}

impl GSSNode {
    pub fn new(level: usize, id: usize, label: usize) -> Self {
        GSSNode {
            level,
            id,
            label,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child_id: usize, edge: usize) {
        self.children.push((child_id, edge));
    }
}

impl fmt::Debug for GSSNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GSSNode(v{}, state={})", self.id, self.label)
    }
}

/// Graph-Structured Stack for GLR parsing
pub struct GSS {
    levels: Vec<Vec<usize>>,
    nodes: Vec<GSSNode>,
    counter: usize,
    /// HashMap for O(1) lookups: (level, label) -> node_id
    label_map: HashMap<(usize, usize), usize>,
}

impl GSS {
    pub fn new() -> Self {
        GSS {
            levels: Vec::new(),
            nodes: Vec::new(),
            counter: 0,
            label_map: HashMap::default(),
        }
    }

    /// Resize the GSS to have n levels
    pub fn resize(&mut self, n: usize) {
        self.levels = vec![Vec::new(); n];
    }

    /// Create a new node with given label at given level
    pub fn create_node(&mut self, label: usize, level: usize) -> usize {
        let id = self.counter;
        self.counter += 1;
        let node = GSSNode::new(level, id, label);
        self.nodes.push(node);
        self.levels[level].push(id);
        self.label_map.insert((level, label), id);
        id
    }

    /// Find a node with given label at given level - O(1) lookup
    pub fn find_node(&self, label: usize, level: usize) -> Option<usize> {
        self.label_map.get(&(level, label)).copied()
    }

    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> &GSSNode {
        &self.nodes[id]
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: usize) -> &mut GSSNode {
        &mut self.nodes[id]
    }

    /// Check if level has any nodes
    pub fn level_is_empty(&self, level: usize) -> bool {
        self.levels.get(level).map(|l| l.is_empty()).unwrap_or(true)
    }

    /// Find all paths of given length from a node
    /// Returns tuples of (edge_labels, destination_node_id)
    pub fn find_paths_with_length(
        &self,
        node_id: usize,
        length: usize,
    ) -> Vec<(Vec<usize>, usize)> {
        let mut results = Vec::new();
        self.dfs_paths(node_id, length, Vec::new(), &mut results);
        results
    }

    fn dfs_paths(
        &self,
        node_id: usize,
        remaining: usize,
        path: Vec<usize>,
        results: &mut Vec<(Vec<usize>, usize)>,
    ) {
        if remaining == 0 {
            results.push((path, node_id));
            return;
        }

        let node = &self.nodes[node_id];
        for &(child_id, edge) in &node.children {
            let mut new_path = path.clone();
            new_path.push(edge);
            self.dfs_paths(child_id, remaining - 1, new_path, results);
        }
    }
}

// ============================================================================
// SPPF Node - for building parse forest
// ============================================================================

/// A node in the Shared Packed Parse Forest
#[derive(Clone)]
pub struct SPPFNode {
    pub id: usize,
    pub label: i64,
    /// Start position in input (-1 for epsilon-SPPF nodes)
    pub start_pos: i32,
    pub children: Vec<SPPFChild>,
}

/// Child type for SPPF nodes
#[derive(Clone, Debug)]
pub enum SPPFChild {
    Node(usize),
    Packing(Vec<usize>),
}

impl SPPFNode {
    pub fn new(id: usize, label: i64, start_pos: i32) -> Self {
        SPPFNode {
            id,
            label,
            start_pos,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child_id: usize) {
        self.children.push(SPPFChild::Node(child_id));
    }

    /// Check if a sequence of nodes already exists as children
    pub fn check_sequence_exists(&self, nodes: &[usize]) -> bool {
        // Check for packing nodes
        let has_packing = self
            .children
            .iter()
            .any(|c| matches!(c, SPPFChild::Packing(_)));

        if has_packing {
            for child in &self.children {
                if let SPPFChild::Packing(edges) = child {
                    if edges == nodes {
                        return true;
                    }
                }
            }
            false
        } else {
            // No packing nodes - check direct children
            let direct_children: Vec<usize> = self
                .children
                .iter()
                .filter_map(|c| match c {
                    SPPFChild::Node(id) => Some(*id),
                    _ => None,
                })
                .collect();
            direct_children == nodes
        }
    }

    /// Add a list of children, handling packing nodes
    pub fn add_children(&mut self, nodes: Vec<usize>) {
        if self.children.is_empty() {
            for node_id in nodes {
                self.children.push(SPPFChild::Node(node_id));
            }
            return;
        }

        // Check if already exists
        if self.check_sequence_exists(&nodes) {
            return;
        }

        // Convert existing direct children to packing if needed
        let has_packing = self
            .children
            .iter()
            .any(|c| matches!(c, SPPFChild::Packing(_)));
        if !has_packing {
            let existing: Vec<usize> = self
                .children
                .iter()
                .filter_map(|c| match c {
                    SPPFChild::Node(id) => Some(*id),
                    _ => None,
                })
                .collect();
            self.children = vec![SPPFChild::Packing(existing)];
        }

        // Add new packing node
        self.children.push(SPPFChild::Packing(nodes));
    }
}

impl fmt::Debug for SPPFNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SPPFNode({}, label={}, pos={})",
            self.id, self.label, self.start_pos
        )
    }
}

// ============================================================================
// Parser SPPF - runtime SPPF for parsing
// ============================================================================

/// SPPF structure for the parser
pub struct ParserSPPF {
    /// All nodes
    nodes: Vec<SPPFNode>,
    /// Counter for node IDs
    counter: usize,
    /// Epsilon-SPPF nodes (pre-built)
    epsilon_sppf: HashMap<usize, usize>,
    /// I mapping from TableGenerator
    i_map: HashMap<NullableLabel, usize>,
}

impl ParserSPPF {
    /// Create a new Parser SPPF with proper epsilon-SPPF structure
    pub fn new_with_grammar(grammar: &Grammar, i_map: HashMap<NullableLabel, usize>) -> Self {
        let nullable = TableGenerator::compute_nullable(grammar);
        
        let mut sppf = ParserSPPF {
            nodes: Vec::new(),
            counter: 0,
            epsilon_sppf: HashMap::default(),
            i_map: i_map.clone(),
        };

        // Create epsilon node (id = 0, label = 0 for plain epsilon)
        let eps_id = sppf.create_epsilon_node(0, -1);
        sppf.epsilon_sppf.insert(0, eps_id);

        // Map from NullableLabel to SPPF node id
        let mut label_to_node: HashMap<NullableLabel, usize> = HashMap::default();
        
        // Step 1: Create nodes for all nullable non-terminals
        let mut sorted_nullable: Vec<u32> = nullable.iter().copied().collect();
        sorted_nullable.sort_unstable();
        
        for nt in sorted_nullable {
            let label = NullableLabel::Single(nt);
            if let Some(&node_id) = i_map.get(&label) {
                let sppf_label = -(( nt as i64) + 1);  // Encode NT as negative
                let id = sppf.create_epsilon_node(sppf_label, -1);
                sppf.epsilon_sppf.insert(node_id, id);
                label_to_node.insert(label, id);
            }
        }
        
        // Step 2: Add children based on grammar productions
        for (&lhs, productions) in &grammar.rules {
            if !nullable.contains(&lhs) {
                continue;
            }
            
            let lhs_label = NullableLabel::Single(lhs);
            let lhs_node_id = match label_to_node.get(&lhs_label) {
                Some(&id) => id,
                None => continue,
            };
            
            for rhs in productions {
                // Empty production - add epsilon child
                if rhs.is_empty() {
                    sppf.get_node_mut(lhs_node_id).children.push(SPPFChild::Node(eps_id));
                }
                // Check if all symbols in RHS are nullable
                else {
                    let all_nullable = rhs.iter().all(|sym| match sym {
                        NumSymbol::NonTerminal(nt_id) => nullable.contains(nt_id),
                        NumSymbol::Terminal(_) => false,
                    });
                    
                    if all_nullable {
                        let nt_ids: Vec<u32> = rhs
                            .iter()
                            .filter_map(|sym| match sym {
                                NumSymbol::NonTerminal(id) => Some(*id),
                                _ => None,
                            })
                            .collect();
                        
                        // Create packing node with all nullable NTs as children
                        let mut child_ids = Vec::new();
                        for nt_id in &nt_ids {
                            let child_label = NullableLabel::Single(*nt_id);
                            if let Some(&child_node_id) = label_to_node.get(&child_label) {
                                child_ids.push(child_node_id);
                            }
                        }
                        if !child_ids.is_empty() {
                            sppf.get_node_mut(lhs_node_id).children.push(SPPFChild::Packing(child_ids));
                        }
                    }
                }
            }
        }
        
        // Step 3: Handle sequence nodes (partial nullable suffixes)
        // These are created for right-nulled positions in productions
        const INTERMEDIATE_THRESHOLD: i64 = i64::MIN / 4;
        for (nullable_label, &node_id) in &i_map {
            if let NullableLabel::Sequence(nt_ids) = nullable_label {
                if label_to_node.contains_key(nullable_label) {
                    continue;  // Already created
                }
                
                // Create intermediate node for this sequence
                // Use a label below INTERMEDIATE_THRESHOLD so it displays as empty string
                let sppf_label = INTERMEDIATE_THRESHOLD - (node_id as i64);
                let id = sppf.create_epsilon_node(sppf_label, -1);
                sppf.epsilon_sppf.insert(node_id, id);
                
                // Add each NT in sequence as a child
                let mut child_ids = Vec::new();
                for nt_id in nt_ids {
                    let child_label = NullableLabel::Single(*nt_id);
                    if let Some(&child_node_id) = label_to_node.get(&child_label) {
                        child_ids.push(child_node_id);
                    }
                }
                if !child_ids.is_empty() {
                    sppf.get_node_mut(id).children.push(SPPFChild::Packing(child_ids));
                }
                
                label_to_node.insert(nullable_label.clone(), id);
            }
        }

        sppf
    }
    
    pub fn new(i_map: HashMap<NullableLabel, usize>) -> Self {
        let mut sppf = ParserSPPF {
            nodes: Vec::new(),
            counter: 0,
            epsilon_sppf: HashMap::default(),
            i_map,
        };

        // Create epsilon node (id = 0, label = 0 for plain epsilon)
        let eps_id = sppf.create_epsilon_node(0, -1);
        sppf.epsilon_sppf.insert(0, eps_id);

        // Create nodes for each entry in i_map and store mappings
        // Map from NullableLabel to actual SPPF node id
        let mut label_to_node: HashMap<NullableLabel, usize> = HashMap::default();
        
        for (nullable_label, &node_id) in &sppf.i_map.clone() {
            if node_id != 0 {
                // Determine the proper label for this epsilon node
                let label = match nullable_label {
                    NullableLabel::Single(nt_id) => {
                        // Single nullable non-terminal: use -(nt_id + 1) encoding
                        -((*nt_id as i64) + 1)
                    }
                    NullableLabel::Sequence(nt_ids) => {
                        // For sequences, we need a special intermediate node label
                        // Use a very large negative number to distinguish from regular NTs
                        // This is a sentinel value that won't conflict with real NT IDs
                        -(1000000 + node_id as i64)
                    }
                    NullableLabel::Epsilon => 0,
                };
                let id = sppf.create_epsilon_node(label, -1);
                sppf.epsilon_sppf.insert(node_id, id);
                label_to_node.insert(nullable_label.clone(), id);
            }
        }

        // Now add children to epsilon nodes based on their structure
        // For Single(nt): add epsilon child
        // For Sequence(nts): add children for each nt in sequence
        for (nullable_label, &sppf_node_id) in &label_to_node {
            match nullable_label {
                NullableLabel::Single(_nt_id) => {
                    // Add epsilon as child
                    sppf.get_node_mut(sppf_node_id).children.push(SPPFChild::Node(eps_id));
                }
                NullableLabel::Sequence(nt_ids) => {
                    // Add each nullable NT in the sequence as a child
                    let mut child_ids = Vec::new();
                    for nt_id in nt_ids {
                        let child_label = NullableLabel::Single(*nt_id);
                        if let Some(&child_node_id) = label_to_node.get(&child_label) {
                            child_ids.push(child_node_id);
                        }
                    }
                    if !child_ids.is_empty() {
                        sppf.get_node_mut(sppf_node_id).children.push(SPPFChild::Packing(child_ids));
                    }
                }
                NullableLabel::Epsilon => {}
            }
        }

        sppf
    }

    fn create_epsilon_node(&mut self, label: i64, start_pos: i32) -> usize {
        let id = self.counter;
        self.counter += 1;
        self.nodes.push(SPPFNode::new(id, label, start_pos));
        id
    }

    /// Create a new SPPF node
    pub fn create_node(&mut self, label: i64, start_pos: i32) -> usize {
        let id = self.counter;
        self.counter += 1;
        self.nodes.push(SPPFNode::new(id, label, start_pos));
        id
    }

    /// Get epsilon-SPPF node by label
    pub fn get_epsilon(&self, label: usize) -> usize {
        *self.epsilon_sppf.get(&label).unwrap_or(&0)
    }

    /// Get node by ID
    pub fn get_node(&self, id: usize) -> &SPPFNode {
        &self.nodes[id]
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: usize) -> &mut SPPFNode {
        &mut self.nodes[id]
    }

    /// Get I mapping for a single non-terminal
    pub fn get_i_single(&self, nt: u32) -> usize {
        *self.i_map.get(&NullableLabel::Single(nt)).unwrap_or(&0)
    }
}

// ============================================================================
// Parsed Action - Action parsed from CSV
// ============================================================================

/// Action parsed from CSV format
#[derive(Debug, Clone)]
pub enum ParsedAction {
    /// Push state k onto stack
    Push(usize),
    /// Reduce: (lhs_nt, dot_position, sppf_label)
    Reduce(i32, usize, usize),
    /// Accept
    Accept,
}

/// Parse an action string from CSV format
fn parse_action(action: &str) -> Option<ParsedAction> {
    if action == "acc" {
        return Some(ParsedAction::Accept);
    }

    let parts: Vec<&str> = action.split('.').collect();
    if parts.is_empty() {
        return None;
    }

    match parts[0] {
        "p" if parts.len() == 2 => parts[1].parse().ok().map(ParsedAction::Push),
        "r" if parts.len() == 4 => {
            let lhs = parts[1].parse::<i32>().ok()?;
            let dot = parts[2].parse::<usize>().ok()?;
            let label = parts[3].parse::<usize>().ok()?;
            Some(ParsedAction::Reduce(lhs, dot, label))
        }
        _ => None,
    }
}

// ============================================================================
// RnglrParser - RNGLR Parser
// ============================================================================

use crate::parse_tree::ParseTree;
use std::io::BufRead;

/// Parse table type: state -> symbol -> list of actions
pub type ParseTable = HashMap<usize, HashMap<i32, Vec<ParsedAction>>>;

/// RNGLR Parser implementation
///
/// This implements the Right-Nulled GLR parsing algorithm based on
/// the pseudocode by Giorgios Robert Economopoulos
pub struct RnglrParser {
    table: ParseTable,
    pub grammar: Option<Grammar>,
    end_of_input: i32,
    accept_states: HashSet<usize>,
    /// I-map for epsilon-SPPF nodes
    i_map: HashMap<NullableLabel, usize>,
}

impl RnglrParser {
    /// Create a new parser from a table
    pub fn new(table: ParseTable) -> Self {
        let accept_states = Self::find_accept_states(&table);
        RnglrParser {
            table,
            grammar: None,
            end_of_input: 0,
            accept_states,
            i_map: HashMap::default(),
        }
    }

    /// Create a new parser with grammar for parse tree construction
    pub fn with_grammar(table: ParseTable, grammar: Grammar) -> Self {
        let accept_states = Self::find_accept_states(&table);
        let i_map = Self::compute_i_map(&grammar);
        RnglrParser {
            table,
            grammar: Some(grammar),
            end_of_input: 0,
            accept_states,
            i_map,
        }
    }

    /// Compute the i_map from a grammar
    fn compute_i_map(grammar: &Grammar) -> HashMap<NullableLabel, usize> {
        // Compute nullable set
        let nullable = TableGenerator::compute_nullable(grammar);
        
        // Build epsilon-SPPF and get i_map
        let sppf = SPPF::new(grammar, &nullable);
        // Convert from standard HashMap to FxHashMap
        sppf.i_map.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// Set the grammar and recompute i_map
    pub fn set_grammar(&mut self, grammar: Grammar) {
        self.i_map = Self::compute_i_map(&grammar);
        self.grammar = Some(grammar);
    }

    /// Import parse table from a CSV file
    ///
    /// CSV format:
    /// - First row: header with "state" followed by symbol columns
    /// - Each subsequent row: state_id, followed by actions for each symbol
    /// - Actions are dot-separated: "p.{state}", "r.{lhs}.{dot}.{label}", "acc"
    /// - Multiple actions are separated by "/"
    pub fn import_table_from_csv<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header
        let header_line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty CSV file"))??;
        let header: Vec<&str> = header_line.split(',').collect();

        // Extract symbol columns (skip "state" column)
        let symbols: Vec<i32> = header[1..]
            .iter()
            .map(|s| s.trim().parse::<i32>().unwrap_or(0))
            .collect();

        let mut table: ParseTable = HashMap::default();

        // Parse each row
        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();

            if parts.is_empty() {
                continue;
            }

            let state_id: usize = parts[0]
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid state ID"))?;

            let mut state_actions: HashMap<i32, Vec<ParsedAction>> = HashMap::default();

            for (idx, &symbol) in symbols.iter().enumerate() {
                let action_str = parts.get(idx + 1).unwrap_or(&"").trim();
                if action_str.is_empty() {
                    state_actions.insert(symbol, Vec::new());
                    continue;
                }

                let actions: Vec<ParsedAction> = action_str
                    .split('/')
                    .filter_map(|a| parse_action(a.trim()))
                    .collect();

                state_actions.insert(symbol, actions);
            }

            table.insert(state_id, state_actions);
        }

        Ok(RnglrParser::new(table))
    }

    /// Find accept states in the table
    fn find_accept_states(table: &ParseTable) -> HashSet<usize> {
        let mut accept_states = HashSet::default();
        let end_of_input = 0i32;

        for (&state_id, actions) in table {
            if let Some(action_list) = actions.get(&end_of_input) {
                for action in action_list {
                    if matches!(action, ParsedAction::Accept) {
                        accept_states.insert(state_id);
                    }
                }
            }
        }

        accept_states
    }

    /// Get actions for a state and symbol
    pub fn get_actions(&self, state: usize, symbol: i32) -> &[ParsedAction] {
        self.table
            .get(&state)
            .and_then(|s| s.get(&symbol))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Parse input tokens (as i32 symbol IDs)
    ///
    /// Returns:
    /// Parse input and return a parse tree
    ///
    /// # Returns
    /// * `Some(tree)` if parsing succeeds
    /// * `None` if parsing fails
    pub fn parse(&self, input: &[i32]) -> Option<ParseTree> {
        // Handle empty input
        if input.is_empty() {
            let actions = self.get_actions(0, self.end_of_input);
            if actions.iter().any(|a| matches!(a, ParsedAction::Accept)) {
                let name = match &self.grammar {
                    Some(g) => g.start_str().unwrap_or("S"),
                    None => "ε",
                };
                return Some(ParseTree::from_str(name, Vec::new()));
            }
            return None;
        }

        // Append end of input marker
        let mut input_with_end: Vec<i32> = input.to_vec();
        input_with_end.push(self.end_of_input);

        // Initialize GSS
        let mut gss = GSS::new();
        gss.resize(input_with_end.len());
        let v_0 = gss.create_node(0, 0); // Initial node with state 0

        // Initialize reduction and shift queues
        // Reductions: (node_id, X, m, f, z_id)
        let mut reductions: Vec<(usize, i32, usize, usize, usize)> = Vec::new();
        let mut shifts: Vec<(usize, usize)> = Vec::new();

        // Initialize SPPF
        let mut sppf = if let Some(ref grammar) = self.grammar {
            ParserSPPF::new_with_grammar(grammar, self.i_map.clone())
        } else {
            ParserSPPF::new(self.i_map.clone())
        };

        let mut set_n: HashMap<(i32, usize), usize> = HashMap::default();

        // Check initial actions
        let a_0 = input_with_end[0];
        for action in self.get_actions(0, a_0) {
            match action {
                ParsedAction::Push(k) => {
                    shifts.push((v_0, *k));
                }
                ParsedAction::Reduce(x, m, f) if *m == 0 => {
                    reductions.push((v_0, *x, 0, *f, sppf.get_epsilon(0)));
                }
                _ => {}
            }
        }

        // Main parsing loop
        for i in 0..input_with_end.len() {
            if gss.level_is_empty(i) {
                continue;
            }

            set_n.clear();

            // Process all reductions
            while let Some((v, x, m, f, y)) = reductions.pop() {
                self.reducer(
                    i,
                    v,
                    x,
                    m,
                    f,
                    y,
                    &mut gss,
                    &mut sppf,
                    &mut reductions,
                    &mut shifts,
                    &mut set_n,
                    &input_with_end,
                );
            }

            // Process shifts
            self.shifter(
                i,
                &mut gss,
                &mut sppf,
                &mut reductions,
                &mut shifts,
                &input_with_end,
            );
        }

        // Check for acceptance
        for &accept_state in &self.accept_states {
            if let Some(node_id) = gss.find_node(accept_state, input_with_end.len() - 1) {
                // Find SPPF root
                let node = gss.get_node(node_id);
                for &(child_id, edge_id) in &node.children {
                    let child = gss.get_node(child_id);
                    if child.label == 0 {
                        // Found the root edge - convert SPPF to ParseTree
                        return Some(self.sppf_to_parse_tree(&sppf, edge_id));
                    }
                }
                // Accepted but no tree (recognizer mode)
                let name = match &self.grammar {
                    Some(g) => g.start_str().unwrap_or("S"),
                    None => "S",
                };
                return Some(ParseTree::from_str(name, Vec::new()));
            }
        }

        None
    }

    /// The reducer operation
    fn reducer(
        &self,
        i: usize,
        v: usize,
        x: i32,
        m: usize,
        f: usize,
        y: usize,
        gss: &mut GSS,
        sppf: &mut ParserSPPF,
        reductions: &mut Vec<(usize, i32, usize, usize, usize)>,
        shifts: &mut Vec<(usize, usize)>,
        set_n: &mut HashMap<(i32, usize), usize>,
        input: &[i32],
    ) {
        // Find paths of length max(0, m-1) from v
        let path_length = if m > 0 { m - 1 } else { 0 };
        let paths = gss.find_paths_with_length(v, path_length);

        for (path, u) in paths {
            let k = gss.get_node(u).label;
            let c = gss.get_node(u).level;

            let z = if m == 0 {
                sppf.get_epsilon(f)
            } else {
                if let Some(&existing) = set_n.get(&(x, c)) {
                    existing
                } else {
                    let new_z = sppf.create_node(x as i64, c as i32);
                    set_n.insert((x, c), new_z);
                    new_z
                }
            };

            // Look up actions for (k, X)
            for action in self.get_actions(k, x) {
                if let ParsedAction::Push(next_state) = action {
                    let w = gss.find_node(*next_state, i);

                    if let Some(w_id) = w {
                        // Node exists
                        let w_node = gss.get_node(w_id);
                        let u_is_child = w_node.children.iter().any(|(cid, _)| *cid == u);

                        if !u_is_child {
                            gss.get_node_mut(w_id).add_child(u, z);

                            if m != 0 {
                                // Add new reductions
                                let a_i = input[i];
                                for action in self.get_actions(*next_state, a_i) {
                                    if let ParsedAction::Reduce(b, t, f2) = action {
                                        if *t != 0 {
                                            reductions.push((u, *b, *t, *f2, z));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Create new node
                        let w_id = gss.create_node(*next_state, i);
                        gss.get_node_mut(w_id).add_child(u, z);

                        let a_i = input[i];
                        for action in self.get_actions(*next_state, a_i) {
                            match action {
                                ParsedAction::Push(k2) => {
                                    shifts.push((w_id, *k2));
                                }
                                ParsedAction::Reduce(b, t, f2) => {
                                    if *t == 0 {
                                        reductions.push((w_id, *b, 0, *f2, sppf.get_epsilon(0)));
                                    } else if m != 0 {
                                        reductions.push((u, *b, *t, *f2, z));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Add children to SPPF node
            if m != 0 {
                let mut node_seq: Vec<usize> = path.into_iter().rev().collect();
                node_seq.push(y);
                if f != 0 {
                    node_seq.push(sppf.get_epsilon(f));
                }
                sppf.get_node_mut(z).add_children(node_seq);
            }
        }
    }

    /// The shifter operation
    pub(crate) fn shifter(
        &self,
        i: usize,
        gss: &mut GSS,
        sppf: &mut ParserSPPF,
        reductions: &mut Vec<(usize, i32, usize, usize, usize)>,
        shifts: &mut Vec<(usize, usize)>,
        input: &[i32],
    ) {
        if i + 1 >= input.len() {
            return;
        }

        let mut new_shifts: Vec<(usize, usize)> = Vec::new();
        let z = sppf.create_node(input[i] as i64, i as i32);

        while let Some((v, k)) = shifts.pop() {
            let node = gss.find_node(k, i + 1);

            if let Some(node_id) = node {
                gss.get_node_mut(node_id).add_child(v, z);

                let a_next = input[i + 1];
                for action in self.get_actions(k, a_next) {
                    if let ParsedAction::Reduce(b, t, f) = action {
                        if *t != 0 {
                            reductions.push((v, *b, *t, *f, z));
                        }
                    }
                }
            } else {
                let new_node = gss.create_node(k, i + 1);
                gss.get_node_mut(new_node).add_child(v, z);

                let a_next = input[i + 1];
                for action in self.get_actions(k, a_next) {
                    match action {
                        ParsedAction::Push(k2) => {
                            new_shifts.push((new_node, *k2));
                        }
                        ParsedAction::Reduce(b, t, f) => {
                            if *t == 0 {
                                reductions.push((new_node, *b, 0, *f, sppf.get_epsilon(0)));
                            } else {
                                reductions.push((v, *b, *t, *f, z));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        *shifts = new_shifts;
    }

    /// Convert SPPF node to ParseTree (Iterative version with cycle detection)
    pub(crate) fn sppf_to_parse_tree(&self, sppf: &ParserSPPF, root_node_id: usize) -> ParseTree {
        // Estimate capacity based on SPPF size to reduce allocations
        let estimated_size = sppf.nodes.len().min(256);
        
        // Track ALL visited nodes to prevent re-processing (like GLL/Earley)
        // Once visited, we skip the node entirely - this prevents infinite loops on cycles
        let mut visited: HashSet<usize> = HashSet::with_capacity_and_hasher(estimated_size, Default::default());
        
        // Stack holds: (NodeID, num_children, current_child_idx, first_child_offset)
        // We store children inline in a separate vec to avoid allocating Vec per node
        let mut stack: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(estimated_size);
        let mut all_children: Vec<usize> = Vec::with_capacity(estimated_size * 2);
        
        // Get root children and push to stack
        let root_node = sppf.get_node(root_node_id);
        let root_child_offset = all_children.len();
        let root_num_children = self.get_flattened_children_inline(root_node, &mut all_children);
        stack.push((root_node_id, root_num_children, 0, root_child_offset));
        visited.insert(root_node_id);
        
        // Results stack holds constructed ParseTree nodes
        let mut results: Vec<ParseTree> = Vec::with_capacity(estimated_size);
        
        // Epsilon node for cycles (reused)
        let epsilon_tree = ParseTree::from_str("ε", vec![]);

        loop {
            if let Some((_, num_children, child_idx, child_offset)) = stack.last_mut() {
                if *child_idx < *num_children {
                    // Get next child
                    let next_child_id = all_children[*child_offset + *child_idx];
                    *child_idx += 1;
                    
                    // Check if already visited - if so, add epsilon placeholder
                    if visited.contains(&next_child_id) {
                        // Already processed this node - add epsilon to avoid duplicates/cycles
                        results.push(epsilon_tree.clone());
                        continue;
                    }
                    
                    visited.insert(next_child_id);
                    
                    // Get children for next node
                    let next_node = sppf.get_node(next_child_id);
                    let next_child_offset = all_children.len();
                    let next_num_children = self.get_flattened_children_inline(next_node, &mut all_children);
                    stack.push((next_child_id, next_num_children, 0, next_child_offset));
                } else {
                    // All children processed for this node. Build the ParseTree.
                    let (finished_id, finished_num_children, _, _) = stack.pop().unwrap();

                    // Pop the results corresponding to this node's children
                    let start_idx = results.len().saturating_sub(finished_num_children);
                    let node_children: Vec<ParseTree> = results.drain(start_idx..).collect();

                    // Determine the name (Label) of the node
                    let node = sppf.get_node(finished_id);
                    let name = self.get_node_name(node.label);

                    results.push(ParseTree::from_str(&name, node_children));
                }
            } else {
                // Stack is empty, we're done
                break;
            }
        }

        results
            .pop()
            .expect("Failed to generate parse tree: stack empty")
    }

    /// Helper to get children IDs inline, flattening packing nodes (mimics original logic)
    /// Returns the number of children added
    #[inline]
    fn get_flattened_children_inline(&self, node: &SPPFNode, ids: &mut Vec<usize>) -> usize {
        let start_len = ids.len();
        for child in &node.children {
            match child {
                SPPFChild::Node(id) => ids.push(*id),
                SPPFChild::Packing(p_ids) => {
                    // For ambiguous parses, the original logic flattened the packing nodes.
                    // We preserve that behavior here.
                    ids.extend_from_slice(p_ids);
                }
            }
        }
        ids.len() - start_len
    }
    
    /// Get node name from label with caching to avoid repeated string allocations
    #[inline]
    fn get_node_name(&self, label: i64) -> String {
        if let Some(grammar) = &self.grammar {
            if label < 0 {
                // Non-terminal
                let nt_id = (-label) as u32;
                grammar
                    .non_terminals
                    .get_str(nt_id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("NT{}", nt_id))
            } else if label > 0 {
                // Terminal
                let t_id = label as u32;
                grammar
                    .terminals
                    .get_str(t_id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{}", label))
            } else {
                "ε".to_string()
            }
        } else {
            format!("{}", label)
        }
    }
    /// Convert SPPF node to ALL possible ParseTrees (for ambiguous grammars)
    /// This enumerates all parse trees represented by the SPPF
    pub(crate) fn sppf_to_all_parse_trees(
        &self,
        sppf: &ParserSPPF,
        node_id: usize,
    ) -> Vec<ParseTree> {
        let mut memo: HashMap<usize, Vec<ParseTree>> = HashMap::default();
        self.sppf_to_all_parse_trees_impl(sppf, node_id, &mut memo)
    }

    /// Internal implementation with memoization
    fn sppf_to_all_parse_trees_impl(
        &self,
        sppf: &ParserSPPF,
        node_id: usize,
        memo: &mut HashMap<usize, Vec<ParseTree>>,
    ) -> Vec<ParseTree> {
        // Check if we've already computed this node
        if let Some(cached) = memo.get(&node_id) {
            return cached.clone();
        }

        let node = sppf.get_node(node_id);

        // Note: Labels are stored with +1 offset for terminals and -1 offset for non-terminals
        // to avoid collision with 0 (end of input)
        // BRNGLR intermediate nodes have labels < INTERMEDIATE_THRESHOLD (very large negative)
        const INTERMEDIATE_THRESHOLD: i64 = i64::MIN / 4;

        let name = if let Some(grammar) = &self.grammar {
            if node.label < INTERMEDIATE_THRESHOLD {
                // BRNGLR intermediate node - use empty string
                "".to_string()
            } else if node.label < 0 {
                // Regular non-terminal: label is -(id + 1), so id = -label - 1
                let nt_id = ((-node.label) - 1) as u32;
                grammar
                    .non_terminals
                    .get_str(nt_id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "".to_string()) // Unknown NT also shown as empty
            } else if node.label > 0 {
                // Terminal: label is id + 1, so id = label - 1
                let t_id = (node.label - 1) as u32;
                grammar
                    .terminals
                    .get_str(t_id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("T{}", t_id))
            } else {
                // label == 0 means epsilon or special node
                "ε".to_string()
            }
        } else {
            format!("{}", node.label)
        };

        // If no children, return a single leaf node
        if node.children.is_empty() {
            let result = vec![ParseTree::from_str(&name, vec![])];
            memo.insert(node_id, result.clone());
            return result;
        }

        // Check if there are packing nodes (ambiguity)
        let has_packing = node
            .children
            .iter()
            .any(|c| matches!(c, SPPFChild::Packing(_)));

        let result = if has_packing {
            // Each packing node represents an alternative derivation
            let mut all_trees = Vec::new();
            for child in &node.children {
                if let SPPFChild::Packing(ids) = child {
                    // Get all combinations of children for this packing node
                    let child_trees: Vec<Vec<ParseTree>> = ids
                        .iter()
                        .map(|id| self.sppf_to_all_parse_trees_impl(sppf, *id, memo))
                        .collect();

                    // Compute cartesian product of all child tree possibilities
                    let combinations = Self::cartesian_product(&child_trees);
                    for combo in combinations {
                        all_trees.push(ParseTree::from_str(&name, combo));
                    }
                }
            }
            all_trees
        } else {
            // No packing nodes - just direct children
            let child_trees: Vec<Vec<ParseTree>> = node
                .children
                .iter()
                .filter_map(|c| match c {
                    SPPFChild::Node(id) => Some(self.sppf_to_all_parse_trees_impl(sppf, *id, memo)),
                    _ => None,
                })
                .collect();

            // Compute cartesian product of all child tree possibilities
            let combinations = Self::cartesian_product(&child_trees);
            combinations
                .into_iter()
                .map(|combo| ParseTree::from_str(&name, combo))
                .collect()
        };

        memo.insert(node_id, result.clone());
        result
    }

    /// Compute cartesian product of vectors
    fn cartesian_product(lists: &[Vec<ParseTree>]) -> Vec<Vec<ParseTree>> {
        if lists.is_empty() {
            return vec![vec![]];
        }

        let mut result = vec![vec![]];
        for list in lists {
            let mut new_result = Vec::new();
            for existing in &result {
                for item in list {
                    let mut new_combo = existing.clone();
                    new_combo.push(item.clone());
                    new_result.push(new_combo);
                }
            }
            result = new_result;
        }
        result
    }

    /// Parse and return ALL parse trees (for ambiguous grammars)
    pub fn parse_all(&self, input: &[i32]) -> Result<Vec<ParseTree>, String> {
        // Handle empty input
        if input.is_empty() {
            let actions = self.get_actions(0, self.end_of_input);
            if actions.iter().any(|a| matches!(a, ParsedAction::Accept)) {
                return Ok(vec![]); // Accept empty input
            }
            return Err("Empty input not accepted".to_string());
        }

        // Append end of input marker
        let mut input_with_end: Vec<i32> = input.to_vec();
        input_with_end.push(self.end_of_input);

        // Initialize GSS
        let mut gss = GSS::new();
        gss.resize(input_with_end.len());
        let v_0 = gss.create_node(0, 0);

        // Initialize reduction and shift queues
        let mut reductions: Vec<(usize, i32, usize, usize, usize)> = Vec::new();
        let mut shifts: Vec<(usize, usize)> = Vec::new();

        // Initialize SPPF
        let mut sppf = if let Some(ref grammar) = self.grammar {
            ParserSPPF::new_with_grammar(grammar, self.i_map.clone())
        } else {
            ParserSPPF::new(self.i_map.clone())
        };

        let mut set_n: HashMap<(i32, usize), usize> = HashMap::default();

        // Check initial actions
        let a_0 = input_with_end[0];
        for action in self.get_actions(0, a_0) {
            match action {
                ParsedAction::Push(k) => {
                    shifts.push((v_0, *k));
                }
                ParsedAction::Reduce(x, m, f) if *m == 0 => {
                    reductions.push((v_0, *x, 0, *f, sppf.get_epsilon(0)));
                }
                _ => {}
            }
        }

        // Main parsing loop
        for i in 0..input_with_end.len() {
            if gss.level_is_empty(i) {
                continue;
            }

            set_n.clear();

            while let Some((v, x, m, f, y)) = reductions.pop() {
                self.reducer(
                    i,
                    v,
                    x,
                    m,
                    f,
                    y,
                    &mut gss,
                    &mut sppf,
                    &mut reductions,
                    &mut shifts,
                    &mut set_n,
                    &input_with_end,
                );
            }

            self.shifter(
                i,
                &mut gss,
                &mut sppf,
                &mut reductions,
                &mut shifts,
                &input_with_end,
            );
        }

        // Check for acceptance and collect all parse trees
        let mut all_trees = Vec::new();
        for &accept_state in &self.accept_states {
            if let Some(node_id) = gss.find_node(accept_state, input_with_end.len() - 1) {
                let node = gss.get_node(node_id);
                for &(child_id, edge_id) in &node.children {
                    let child = gss.get_node(child_id);
                    if child.label == 0 {
                        // Found the root edge - get all trees from this SPPF node
                        let trees = self.sppf_to_all_parse_trees(&sppf, edge_id);
                        all_trees.extend(trees);
                    }
                }
            }
        }

        if all_trees.is_empty() {
            Err("Input not accepted".to_string())
        } else {
            Ok(all_trees)
        }
    }

    /// Check if this is an accepting parse
    pub fn accepts(&self, input: &[i32]) -> bool {
        self.parse(input).is_some()
    }
}

// ============================================================================
// BrnglrParser - Binary RNGLR Parser
// ============================================================================

/// BRNGLR Parser implementation
///
/// This extends RNGLR with binary reduction decomposition.
/// For reductions of length > 2, it breaks them into binary steps
/// to achieve better worst-case complexity (O(n^3) instead of O(n^(p+1))).
pub struct BrnglrParser {
    /// The underlying RNGLR parser (shares table, grammar, etc.)
    inner: RnglrParser,
}

impl BrnglrParser {
    /// Create a new BRNGLR parser from a table
    pub fn new(table: ParseTable) -> Self {
        BrnglrParser {
            inner: RnglrParser::new(table),
        }
    }

    /// Create a new parser with grammar for parse tree construction
    pub fn with_grammar(table: ParseTable, grammar: Grammar) -> Self {
        BrnglrParser {
            inner: RnglrParser::with_grammar(table, grammar),
        }
    }

    /// Import parse table from a CSV file
    pub fn import_table_from_csv<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let inner = RnglrParser::import_table_from_csv(path)?;
        Ok(BrnglrParser { inner })
    }

    /// Set the grammar and recompute i_map
    pub fn set_grammar(&mut self, grammar: Grammar) {
        self.inner.set_grammar(grammar);
    }

    /// Check if this is an accepting parse
    pub fn accepts(&self, input: &[i32]) -> bool {
        self.parse(input).is_some()
    }

    /// Parse input tokens using BRNGLR algorithm
    /// Returns the first parse tree if successful, None otherwise
    pub fn parse(&self, input: &[i32]) -> Option<ParseTree> {
        // Handle empty input
        if input.is_empty() {
            let actions = self.inner.get_actions(0, self.inner.end_of_input);
            if actions.iter().any(|a| matches!(a, ParsedAction::Accept)) {
                let name = match &self.inner.grammar {
                    Some(g) => g.start_str().unwrap_or("S"),
                    None => "ε",
                };
                return Some(ParseTree::from_str(name, Vec::new()));
            }
            return None;
        }

        // Append end of input marker
        let mut input_with_end: Vec<i32> = input.to_vec();
        input_with_end.push(self.inner.end_of_input);

        // Initialize GSS
        let mut gss = GSS::new();
        gss.resize(input_with_end.len());
        let v_0 = gss.create_node(0, 0); // Initial node with state 0

        // Initialize reduction and shift queues
        // Reductions: (node_id, X, m, f, z_id)
        let mut reductions: Vec<(usize, i32, usize, usize, usize)> = Vec::new();
        let mut shifts: Vec<(usize, usize)> = Vec::new();

        // Initialize SPPF
        let mut sppf = if let Some(ref grammar) = self.inner.grammar {
            ParserSPPF::new_with_grammar(grammar, self.inner.i_map.clone())
        } else {
            ParserSPPF::new(self.inner.i_map.clone())
        };

        let mut set_n: HashMap<(i32, usize), usize> = HashMap::default();

        // Check initial actions
        let a_0 = input_with_end[0];
        for action in self.inner.get_actions(0, a_0) {
            match action {
                ParsedAction::Push(k) => {
                    shifts.push((v_0, *k));
                }
                ParsedAction::Reduce(x, m, f) if *m == 0 => {
                    reductions.push((v_0, *x, 0, *f, sppf.get_epsilon(0)));
                }
                _ => {}
            }
        }

        // Main parsing loop
        for i in 0..input_with_end.len() {
            if gss.level_is_empty(i) {
                continue;
            }

            set_n.clear();

            // Process all reductions
            while let Some((v, x, m, f, y)) = reductions.pop() {
                self.brnglr_reducer(
                    i,
                    v,
                    x,
                    m,
                    f,
                    y,
                    &mut gss,
                    &mut sppf,
                    &mut reductions,
                    &mut shifts,
                    &mut set_n,
                    &input_with_end,
                );
            }

            // Process shifts
            self.inner.shifter(
                i,
                &mut gss,
                &mut sppf,
                &mut reductions,
                &mut shifts,
                &input_with_end,
            );
        }

        // Check for acceptance
        for &accept_state in &self.inner.accept_states {
            if let Some(node_id) = gss.find_node(accept_state, input_with_end.len() - 1) {
                // Find SPPF root
                let node = gss.get_node(node_id);
                for &(child_id, edge_id) in &node.children {
                    let child = gss.get_node(child_id);
                    if child.label == 0 {
                        // Found the root edge - convert SPPF to ParseTree
                        return Some(self.inner.sppf_to_parse_tree(&sppf, edge_id));
                    }
                }
                // Accepted but no tree (recognizer mode)
                let name = match &self.inner.grammar {
                    Some(g) => g.start_str().unwrap_or("S"),
                    None => "S",
                };
                return Some(ParseTree::from_str(name, Vec::new()));
            }
        }

        None
    }

    /// Parse and return ALL parse trees (for ambiguous grammars)
    pub fn parse_all(&self, input: &[i32]) -> Result<Vec<ParseTree>, String> {
        // Handle empty input
        if input.is_empty() {
            let actions = self.inner.get_actions(0, self.inner.end_of_input);
            if actions.iter().any(|a| matches!(a, ParsedAction::Accept)) {
                return Ok(vec![]); // Accept empty input
            }
            return Err("Empty input not accepted".to_string());
        }

        // Append end of input marker
        let mut input_with_end: Vec<i32> = input.to_vec();
        input_with_end.push(self.inner.end_of_input);

        // Initialize GSS
        let mut gss = GSS::new();
        gss.resize(input_with_end.len());
        let v_0 = gss.create_node(0, 0);

        // Initialize reduction and shift queues
        let mut reductions: Vec<(usize, i32, usize, usize, usize)> = Vec::new();
        let mut shifts: Vec<(usize, usize)> = Vec::new();

        // Initialize SPPF
        let mut sppf = if let Some(ref grammar) = self.inner.grammar {
            ParserSPPF::new_with_grammar(grammar, self.inner.i_map.clone())
        } else {
            ParserSPPF::new(self.inner.i_map.clone())
        };

        let mut set_n: HashMap<(i32, usize), usize> = HashMap::default();

        // Check initial actions
        let a_0 = input_with_end[0];
        for action in self.inner.get_actions(0, a_0) {
            match action {
                ParsedAction::Push(k) => {
                    shifts.push((v_0, *k));
                }
                ParsedAction::Reduce(x, m, f) if *m == 0 => {
                    reductions.push((v_0, *x, 0, *f, sppf.get_epsilon(0)));
                }
                _ => {}
            }
        }

        // Main parsing loop
        for i in 0..input_with_end.len() {
            if gss.level_is_empty(i) {
                continue;
            }

            set_n.clear();

            while let Some((v, x, m, f, y)) = reductions.pop() {
                self.brnglr_reducer(
                    i,
                    v,
                    x,
                    m,
                    f,
                    y,
                    &mut gss,
                    &mut sppf,
                    &mut reductions,
                    &mut shifts,
                    &mut set_n,
                    &input_with_end,
                );
            }

            self.inner.shifter(
                i,
                &mut gss,
                &mut sppf,
                &mut reductions,
                &mut shifts,
                &input_with_end,
            );
        }

        // Check for acceptance and collect all parse trees
        let mut all_trees = Vec::new();
        for &accept_state in &self.inner.accept_states {
            if let Some(node_id) = gss.find_node(accept_state, input_with_end.len() - 1) {
                let node = gss.get_node(node_id);
                for &(child_id, edge_id) in &node.children {
                    let child = gss.get_node(child_id);
                    if child.label == 0 {
                        // Found the root edge - get all trees from this SPPF node
                        let trees = self.inner.sppf_to_all_parse_trees(&sppf, edge_id);
                        all_trees.extend(trees);
                    }
                }
            }
        }

        if all_trees.is_empty() {
            Err("Input not accepted".to_string())
        } else {
            Ok(all_trees)
        }
    }

    /// BRNGLR reducer - handles reductions with binary decomposition
    ///
    /// For m <= 2: same as RNGLR
    /// For m > 2: uses intermediate nodes X_m to decompose into binary reductions
    fn brnglr_reducer(
        &self,
        i: usize,
        v: usize,
        x: i32,
        m: usize,
        f: usize,
        y: usize,
        gss: &mut GSS,
        sppf: &mut ParserSPPF,
        reductions: &mut Vec<(usize, i32, usize, usize, usize)>,
        shifts: &mut Vec<(usize, usize)>,
        set_n: &mut HashMap<(i32, usize), usize>,
        input: &[i32],
    ) {
        // Build X_: for m >= 2, get children of v; otherwise use (v, epsilon)
        let x_prime: Vec<(usize, usize)> = if m >= 2 {
            gss.get_node(v).children.clone()
        } else {
            vec![(v, sppf.get_epsilon(0))]
        };

        if m <= 2 {
            // Same as RNGLR for small reductions
            for (u, x_edge) in x_prime {
                let k = gss.get_node(u).label;
                let c = gss.get_node(u).level;

                let z = if m == 0 {
                    sppf.get_epsilon(f)
                } else {
                    if let Some(&existing) = set_n.get(&(x, c)) {
                        existing
                    } else {
                        let new_z = sppf.create_node(x as i64, c as i32);
                        set_n.insert((x, c), new_z);
                        new_z
                    }
                };

                // Look up actions for (k, X)
                for action in self.inner.get_actions(k, x) {
                    if let ParsedAction::Push(next_state) = action {
                        let w = gss.find_node(*next_state, i);

                        if let Some(w_id) = w {
                            let w_node = gss.get_node(w_id);
                            let u_is_child = w_node.children.iter().any(|(cid, _)| *cid == u);

                            if !u_is_child {
                                gss.get_node_mut(w_id).add_child(u, z);

                                if m != 0 {
                                    let a_i = input[i];
                                    for action in self.inner.get_actions(*next_state, a_i) {
                                        if let ParsedAction::Reduce(b, t, f2) = action {
                                            if *t != 0 {
                                                reductions.push((u, *b, *t, *f2, z));
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            let w_id = gss.create_node(*next_state, i);
                            gss.get_node_mut(w_id).add_child(u, z);

                            let a_i = input[i];
                            for action in self.inner.get_actions(*next_state, a_i) {
                                match action {
                                    ParsedAction::Push(k2) => {
                                        shifts.push((w_id, *k2));
                                    }
                                    ParsedAction::Reduce(b, t, f2) => {
                                        if *t == 0 {
                                            reductions.push((
                                                w_id,
                                                *b,
                                                0,
                                                *f2,
                                                sppf.get_epsilon(0),
                                            ));
                                        } else if m != 0 {
                                            reductions.push((u, *b, *t, *f2, z));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                // Add children to SPPF node
                if m != 0 {
                    let mut node_seq: Vec<usize> = if m == 1 { vec![y] } else { vec![x_edge, y] };
                    if f != 0 {
                        node_seq.push(sppf.get_epsilon(f));
                    }
                    sppf.get_node_mut(z).add_children(node_seq);
                }
            }
        } else {
            // m > 2: Binary reduction decomposition
            // Use intermediate node X_m to break into smaller reductions

            // Encode X_m as a special label using a very large negative offset
            // to guarantee no collision with real non-terminals (which use small negative numbers)
            // Formula: INTERMEDIATE_BASE - ((-x) * MAX_PRODUCTION_LENGTH + m)
            // This ensures labels are far below any realistic grammar symbol
            const INTERMEDIATE_BASE: i64 = i64::MIN / 2;
            const MAX_PRODUCTION_LENGTH: i64 = 1000; // Max symbols in a production
            let x_m_label = INTERMEDIATE_BASE - ((-x as i64) * MAX_PRODUCTION_LENGTH + (m as i64));
            let x_m_state = (i32::MAX / 2) as usize + ((-x) as usize) * 100 + m;

            // Find or create node for X_m at level i
            let w = gss.find_node(x_m_state, i);
            let w_id = w.unwrap_or_else(|| gss.create_node(x_m_state, i));

            for (u, x_edge) in x_prime {
                // Check if u is already a child of w
                let existing_edge = {
                    let w_node = gss.get_node(w_id);
                    w_node
                        .children
                        .iter()
                        .find(|(cid, _)| *cid == u)
                        .map(|(_, e)| *e)
                };

                let z = if let Some(edge_id) = existing_edge {
                    edge_id
                } else {
                    // Create new SPPF node for intermediate reduction
                    let c = gss.get_node(u).level;
                    let new_z = sppf.create_node(x_m_label, c as i32);
                    gss.get_node_mut(w_id).add_child(u, new_z);

                    // Add reduction for m-1 (continue decomposition)
                    reductions.push((u, x, m - 1, 0, new_z));

                    new_z
                };

                // Add children to SPPF node: [x_edge, y] and optionally epsilon
                let mut node_seq = vec![x_edge, y];
                if f != 0 {
                    node_seq.push(sppf.get_epsilon(f));
                }
                sppf.get_node_mut(z).add_children(node_seq);
            }
        }
    }
}

// =================================== End of Code =======================================





















// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::{self, simple_grammar};

    // #[test]
    // fn test_table_generator_creation() {
    //     let grammar = simple_grammar();
    //     let table_gen = TableGenerator::new(&grammar);

    //     assert!(!table_gen.nullable.is_empty() || table_gen.nullable.is_empty()); // Either is valid
    //     assert!(!table_gen.first.is_empty());
    //     println!("Nullable: {:?}", table_gen.nullable);
    //     println!("FIRST: {:?}", table_gen.first);
    // }
    #[test]
    fn test_glr_epsilon() {
        let json = r#"{
            "name": "nested",
            "start": "<S>",
            "rules": {
                "<S>": [["a", "<B>", "<B>", "<C>"]],
                "<B>": [["b"], []],
                "<C>": [["<D>"]],
                "<D>": [[]]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let table_gen = TableGenerator::new(&grammar);

        // Export to CSV (using numeric format for import compatibility)
        let temp_path = "/tmp/glr_epsilon_test.csv";
        table_gen
            .export_to_csv_numeric(temp_path)
            .expect("Failed to export CSV");
        println!("Exported GLR parse table to {}", temp_path);

        let input = "ab";
        let tokens = grammar.tokenize(input).expect("Failed to tokenize input");
        let glr_tokens: Vec<_> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        // Import table and add grammar for proper symbol display
        let mut parser = RnglrParser::import_table_from_csv(temp_path)
            .expect("Failed to import CSV into parser");
        parser.set_grammar(grammar);
        
        let parse_trees = parser
            .parse_all(&glr_tokens)
            .expect("Failed to parse input");

        for (i, tree) in parse_trees.iter().enumerate() {
            println!("Parse Tree {}:\n{}", i + 1, tree.display());
        }
    }

    #[test]
    fn test_state_generation() {
        let grammar = simple_grammar();
        let table_gen = TableGenerator::new(&grammar);
        let (states, goto_map) = table_gen.generate_states();

        println!("Number of states: {}", states.len());
        println!("Number of transitions: {}", goto_map.len());

        for (state_id, items) in &states {
            println!("State {}:", state_id);
            for item in items {
                println!("  {:?}", item);
            }
        }

        assert!(!states.is_empty());
    }

    #[test]
    fn test_parse_table_generation() {
        let grammar = simple_grammar();
        let table_gen = TableGenerator::new(&grammar);
        let table = table_gen.generate_parse_table();

        println!("Parse table has {} states", table.len());

        for (state_id, actions) in &table {
            println!("State {}:", state_id);
            for (symbol, action_list) in actions {
                if !action_list.is_empty() {
                    println!("  {:?} -> {:?}", symbol, action_list);
                }
            }
        }

        assert!(!table.is_empty());
    }

    #[test]
    fn test_export_to_csv() {
        let grammar = simple_grammar();
        let table_gen = TableGenerator::new(&grammar);

        // Export to a temp file
        let temp_path = "/tmp/glr_test_table.csv";
        table_gen
            .export_to_csv(temp_path)
            .expect("Failed to export CSV");

        // Read and print the file contents
        let contents = std::fs::read_to_string(temp_path).expect("Failed to read CSV");
        println!("CSV contents:\n{}", contents);

        // Verify the file has content
        assert!(!contents.is_empty());
        assert!(contents.contains("state")); // Header should contain "state"

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_json_csv() {
        let grammar =
            grammars::load_grammar_from_file("grammars/json.json").expect("Failed to load grammar");
        let table_gen = TableGenerator::new(&grammar);
        let table_path = "table/json_glr_table.csv";
        table_gen
            .export_to_csv(table_path)
            .expect("Failed to export JSON CSV");
        println!("Exported JSON grammar parse table to {}", table_path);
    }

    #[test]
    fn test_rnglr_parser_import_csv() {
        // First export a table to CSV
        let grammar = simple_grammar();
        let table_gen = TableGenerator::new(&grammar);

        let temp_path = "/tmp/glr_import_test.csv";
        table_gen
            .export_to_csv_numeric(temp_path)
            .expect("Failed to export CSV");

        // Read and print CSV contents
        let contents: String = std::fs::read_to_string(temp_path).expect("Failed to read CSV");
        println!("CSV to import:\n{}", contents);

        // Import the table
        let parser: RnglrParser =
            RnglrParser::import_table_from_csv(temp_path).expect("Failed to import CSV");

        // Verify the table was imported
        assert!(!parser.table.is_empty());
        println!("Imported {} states", parser.table.len());
        println!("Accept states: {:?}", parser.accept_states);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_rnglr_parser_simple() {
        // Create a simple grammar and table
        let grammar = simple_grammar();
        let table_gen = TableGenerator::new(&grammar);

        // Export to numeric CSV
        let temp_path = "/tmp/glr_parser_test.csv";
        table_gen
            .export_to_csv_numeric(temp_path)
            .expect("Failed to export CSV");

        println!("CSV contents:");
        let contents = std::fs::read_to_string(temp_path).expect("Failed to read CSV");
        println!("{}", contents);

        // Import and create parser
        let parser = RnglrParser::import_table_from_csv(temp_path).expect("Failed to import CSV");

        // The simple_grammar has:
        // S -> "b" A "a" | "b" B "a"
        // A -> "b"
        // B -> "a"
        // Terminals: a=0, b=1
        // So valid inputs are: "b b a" or "b a a"

        // Test with input "b b a" (terminal IDs: 1, 1, 0)
        // Actually we need to check what the actual terminal mappings are
        println!("Testing parser...");

        // For now, just verify the parser was created
        assert!(!parser.table.is_empty());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_rnglr_accepts() {
        // Create a simple test with a manually constructed table
        let mut table: ParseTable = HashMap::default();

        // Simple grammar: S -> a
        // State 0: on 'a' (terminal 1), shift to state 1
        // State 1: on $ (0), accept
        let mut state0: HashMap<i32, Vec<ParsedAction>> = HashMap::default();
        state0.insert(1, vec![ParsedAction::Push(1)]); // On 'a', shift to 1
        state0.insert(0, vec![]); // On $, nothing
        table.insert(0, state0);

        let mut state1: HashMap<i32, Vec<ParsedAction>> = HashMap::default();
        state1.insert(0, vec![ParsedAction::Accept]); // On $, accept
        state1.insert(1, vec![]); // On 'a', nothing
        table.insert(1, state1);

        let parser = RnglrParser::new(table);

        // Input: [1] (just 'a')
        let result = parser.parse(&[1]);
        println!("Parse result: {:?}", result);
        assert!(result.is_some());

        // Invalid input: empty
        let result_empty = parser.parse(&[]);
        println!("Empty input result: {:?}", result_empty);

        // Invalid input: [2] (unknown symbol)
        let result_invalid = parser.parse(&[2]);
        println!("Invalid input result: {:?}", result_invalid);
    }


    #[test]
    fn test_glr_ambiguous_grammar() {
        // Load the ambiguous grammar: S -> S S | a
        // For input "aaa", there should be 2 parse trees:
        //   1. S -> S S where first S -> a, second S -> S S -> a a
        //   2. S -> S S where first S -> S S -> a a, second S -> a

        let grammar = grammars::load_grammar_from_file("grammars/ambi.json")
            .expect("Failed to load ambiguous grammar");

        println!("=== Ambiguous Grammar Test ===");
        println!("Grammar: S -> S S | a");
        println!();

        // Generate the parse table
        let table_gen = TableGenerator::new(&grammar);

        println!("Number of states: {}", table_gen.state_count());
        println!("Has conflicts: {}", table_gen.has_conflicts());
        println!();

        // Export to CSV and import for parsing
        let temp_path = "/tmp/glr_ambi_table.csv";
        table_gen
            .export_to_csv_numeric(temp_path)
            .expect("Failed to export CSV");

        // Print the CSV for debugging
        let contents = std::fs::read_to_string(temp_path).expect("Failed to read CSV");
        println!("Parse table (CSV):\n{}", contents);

        // Create parser with grammar for tree construction
        let parse_table = table_gen.generate_parse_table();
        let mut parser_table: ParseTable = HashMap::default();

        for (&state, actions) in &parse_table {
            let mut state_actions: HashMap<i32, Vec<ParsedAction>> = HashMap::default();
            for (&symbol, action_list) in actions {
                let symbol_i32 = match symbol {
                    NumSymbol::Terminal(id) if id == END_OF_INPUT => 0,
                    NumSymbol::Terminal(id) => (id + 1) as i32,
                    NumSymbol::NonTerminal(id) => -((id + 1) as i32),
                };

                let parsed_actions: Vec<ParsedAction> = action_list
                    .iter()
                    .map(|a| match a {
                        Action::Shift(s) => ParsedAction::Push(*s),
                        Action::Reduce(lhs, dot, label) => {
                            ParsedAction::Reduce(-((*lhs + 1) as i32), *dot, *label)
                        }
                        Action::Accept => ParsedAction::Accept,
                    })
                    .collect();

                state_actions.insert(symbol_i32, parsed_actions);
            }
            parser_table.insert(state, state_actions);
        }

        let parser = RnglrParser::with_grammar(parser_table, grammar.clone());

        // Get terminal ID for 'a'
        let a_id = grammar
            .terminals
            .get_id("a")
            .expect("Terminal 'a' not found");
        println!("Terminal 'a' has ID: {} (parser uses {})", a_id, a_id + 1);
        println!();

        // Test with "aa" (should have 1 parse tree)
        println!("=== Testing input: aa ===");
        let input_aa: Vec<i32> = vec![(a_id + 1) as i32; 2];
        match parser.parse_all(&input_aa) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
        println!();

        // Test with "aaa" (should have 2 parse trees - Catalan number C(2) = 2)
        println!("=== Testing input: aaa ===");
        let input_aaa: Vec<i32> = vec![(a_id + 1) as i32; 3];
        match parser.parse_all(&input_aaa) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
        println!();

        // Test with "aaaa" (should have 5 parse trees - Catalan number C(3) = 5)
        println!("=== Testing input: aaaa ===");
        let input_aaaa: Vec<i32> = vec![(a_id + 1) as i32; 4];
        match parser.parse_all(&input_aaaa) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_brnglr_ambiguous_grammar() {
        // Test BRNGLR with the same ambiguous grammar
        // BRNGLR should produce the same results as RNGLR

        let grammar = grammars::load_grammar_from_file("grammars/ambi.json")
            .expect("Failed to load ambiguous grammar");

        println!("=== BRNGLR Ambiguous Grammar Test ===");
        println!("Grammar: S -> S S | a");
        println!();

        // Generate the parse table
        let table_gen = TableGenerator::new(&grammar);

        // Create parser table for BRNGLR
        let parse_table = table_gen.generate_parse_table();
        let mut parser_table: ParseTable = HashMap::default();

        for (&state, actions) in &parse_table {
            let mut state_actions: HashMap<i32, Vec<ParsedAction>> = HashMap::default();
            for (&symbol, action_list) in actions {
                let symbol_i32 = match symbol {
                    NumSymbol::Terminal(id) if id == END_OF_INPUT => 0,
                    NumSymbol::Terminal(id) => (id + 1) as i32,
                    NumSymbol::NonTerminal(id) => -((id + 1) as i32),
                };

                let parsed_actions: Vec<ParsedAction> = action_list
                    .iter()
                    .map(|a| match a {
                        Action::Shift(s) => ParsedAction::Push(*s),
                        Action::Reduce(lhs, dot, label) => {
                            ParsedAction::Reduce(-((*lhs + 1) as i32), *dot, *label)
                        }
                        Action::Accept => ParsedAction::Accept,
                    })
                    .collect();

                state_actions.insert(symbol_i32, parsed_actions);
            }
            parser_table.insert(state, state_actions);
        }

        let parser = BrnglrParser::with_grammar(parser_table, grammar.clone());

        // Get terminal ID for 'a'
        let a_id = grammar
            .terminals
            .get_id("a")
            .expect("Terminal 'a' not found");
        println!("Terminal 'a' has ID: {} (parser uses {})", a_id, a_id + 1);
        println!();

        // Test with "aa" (should have 1 parse tree)
        println!("=== BRNGLR Testing input: aa ===");
        let input_aa: Vec<i32> = vec![(a_id + 1) as i32; 2];
        match parser.parse_all(&input_aa) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
        println!();

        // Test with "aaa" (should have 2 parse trees)
        println!("=== BRNGLR Testing input: aaa ===");
        let input_aaa: Vec<i32> = vec![(a_id + 1) as i32; 3];
        match parser.parse_all(&input_aaa) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
        println!();

        // Test with "aaaa" (should have 5 parse trees)
        println!("=== BRNGLR Testing input: aaaa ===");
        let input_aaaa: Vec<i32> = vec![(a_id + 1) as i32; 4];
        match parser.parse_all(&input_aaaa) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
                // Verify count matches RNGLR
                assert_eq!(trees.len(), 5, "BRNGLR should find 5 parse trees for aaaa");
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    #[test]
    fn test_brnglr_long_productions() {
        // Test BRNGLR with a grammar that has long productions (3+ symbols on RHS)
        // This exercises the binary reduction decomposition (m > 2 case)
        //
        // Grammar:
        //   S -> A B C D    (4 symbols - will be decomposed)
        //   S -> A B C      (3 symbols - will be decomposed)
        //   A -> a
        //   B -> b
        //   C -> c
        //   D -> d

        println!("=== BRNGLR Long Productions Test ===");
        println!("Grammar with productions of length 3 and 4");
        println!();

        // Create grammar manually
        let mut grammar = Grammar::new("long_productions");

        // Add terminals: a=0, b=1, c=2, d=3
        let a_id = grammar.terminals.get_or_insert("a");
        let b_id = grammar.terminals.get_or_insert("b");
        let c_id = grammar.terminals.get_or_insert("c");
        let d_id = grammar.terminals.get_or_insert("d");

        // Add non-terminals: S=0, A=1, B=2, C=3, D=4
        let s_id = grammar.non_terminals.get_or_insert("<S>");
        let a_nt_id = grammar.non_terminals.get_or_insert("<A>");
        let b_nt_id = grammar.non_terminals.get_or_insert("<B>");
        let c_nt_id = grammar.non_terminals.get_or_insert("<C>");
        let d_nt_id = grammar.non_terminals.get_or_insert("<D>");

        grammar.start = s_id;

        // S -> A B C D (length 4)
        grammar
            .rules
            .entry(s_id)
            .or_insert_with(Vec::new)
            .push(vec![
                NumSymbol::NonTerminal(a_nt_id),
                NumSymbol::NonTerminal(b_nt_id),
                NumSymbol::NonTerminal(c_nt_id),
                NumSymbol::NonTerminal(d_nt_id),
            ]);

        // S -> A B C (length 3)
        grammar
            .rules
            .entry(s_id)
            .or_insert_with(Vec::new)
            .push(vec![
                NumSymbol::NonTerminal(a_nt_id),
                NumSymbol::NonTerminal(b_nt_id),
                NumSymbol::NonTerminal(c_nt_id),
            ]);

        // A -> a
        grammar
            .rules
            .entry(a_nt_id)
            .or_insert_with(Vec::new)
            .push(vec![NumSymbol::Terminal(a_id)]);

        // B -> b
        grammar
            .rules
            .entry(b_nt_id)
            .or_insert_with(Vec::new)
            .push(vec![NumSymbol::Terminal(b_id)]);

        // C -> c
        grammar
            .rules
            .entry(c_nt_id)
            .or_insert_with(Vec::new)
            .push(vec![NumSymbol::Terminal(c_id)]);

        // D -> d
        grammar
            .rules
            .entry(d_nt_id)
            .or_insert_with(Vec::new)
            .push(vec![NumSymbol::Terminal(d_id)]);

        println!("S -> A B C D  (4 non-terminals)");
        println!("S -> A B C    (3 non-terminals)");
        println!("A -> a, B -> b, C -> c, D -> d");
        println!();

        // Generate parse table
        let table_gen = TableGenerator::new(&grammar);
        println!("Number of states: {}", table_gen.state_count());
        println!("Has conflicts: {}", table_gen.has_conflicts());
        println!();

        // Convert parse table for parser
        let parse_table = table_gen.generate_parse_table();
        let mut parser_table: ParseTable = HashMap::default();

        for (&state, actions) in &parse_table {
            let mut state_actions: HashMap<i32, Vec<ParsedAction>> = HashMap::default();
            for (&symbol, action_list) in actions {
                let symbol_i32 = match symbol {
                    NumSymbol::Terminal(id) if id == END_OF_INPUT => 0,
                    NumSymbol::Terminal(id) => (id + 1) as i32,
                    NumSymbol::NonTerminal(id) => -((id + 1) as i32),
                };

                let parsed_actions: Vec<ParsedAction> = action_list
                    .iter()
                    .map(|a| match a {
                        Action::Shift(s) => ParsedAction::Push(*s),
                        Action::Reduce(lhs, dot, label) => {
                            ParsedAction::Reduce(-((*lhs + 1) as i32), *dot, *label)
                        }
                        Action::Accept => ParsedAction::Accept,
                    })
                    .collect();

                state_actions.insert(symbol_i32, parsed_actions);
            }
            parser_table.insert(state, state_actions);
        }

        // Test with BRNGLR
        let brnglr_parser = BrnglrParser::with_grammar(parser_table.clone(), grammar.clone());

        // Test "abcd" - should match S -> A B C D
        println!("=== BRNGLR Testing input: abcd ===");
        let input_abcd: Vec<i32> = vec![
            (a_id + 1) as i32,
            (b_id + 1) as i32,
            (c_id + 1) as i32,
            (d_id + 1) as i32,
        ];
        match brnglr_parser.parse_all(&input_abcd) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
                assert_eq!(trees.len(), 1, "Should have exactly 1 parse tree for abcd");
            }
            Err(e) => {
                println!("Parse error: {}", e);
                panic!("BRNGLR should accept 'abcd'");
            }
        }
        println!();

        // Test "abc" - should match S -> A B C
        println!("=== BRNGLR Testing input: abc ===");
        let input_abc: Vec<i32> = vec![(a_id + 1) as i32, (b_id + 1) as i32, (c_id + 1) as i32];
        match brnglr_parser.parse_all(&input_abc) {
            Ok(trees) => {
                println!("Found {} parse tree(s):", trees.len());
                for (i, tree) in trees.iter().enumerate() {
                    println!("\nParse tree {}:", i + 1);
                    println!("{}", tree.display());
                }
                assert_eq!(trees.len(), 1, "Should have exactly 1 parse tree for abc");
            }
            Err(e) => {
                println!("Parse error: {}", e);
                panic!("BRNGLR should accept 'abc'");
            }
        }
        println!();

        // Compare with RNGLR to ensure same results
        let rnglr_parser = RnglrParser::with_grammar(parser_table, grammar);

        println!("=== Comparing RNGLR vs BRNGLR ===");

        let rnglr_abcd = rnglr_parser.parse_all(&input_abcd);
        let brnglr_abcd = brnglr_parser.parse_all(&input_abcd);

        match (&rnglr_abcd, &brnglr_abcd) {
            (Ok(r), Ok(b)) => {
                println!("RNGLR found {} trees for 'abcd'", r.len());
                println!("BRNGLR found {} trees for 'abcd'", b.len());
                assert_eq!(
                    r.len(),
                    b.len(),
                    "RNGLR and BRNGLR should find same number of trees"
                );
            }
            _ => panic!("Both parsers should succeed"),
        }

        let rnglr_abc = rnglr_parser.parse_all(&input_abc);
        let brnglr_abc = brnglr_parser.parse_all(&input_abc);

        match (&rnglr_abc, &brnglr_abc) {
            (Ok(r), Ok(b)) => {
                println!("RNGLR found {} trees for 'abc'", r.len());
                println!("BRNGLR found {} trees for 'abc'", b.len());
                assert_eq!(
                    r.len(),
                    b.len(),
                    "RNGLR and BRNGLR should find same number of trees"
                );
            }
            _ => panic!("Both parsers should succeed"),
        }

        println!("\n=== BRNGLR correctly handles long productions! ===");
    }

    #[test]
    fn test_rnglr_tinyc_grammar() {
        // Test with TinyC grammar - verifies that CSV import/export works correctly
        let grammar = grammars::load_grammar_from_file("grammars/tinyc.json")
            .expect("Failed to load TinyC grammar");

        // Generate the parse table
        let table_gen = TableGenerator::new(&grammar);

        // Export to CSV and import for parsing
        let temp_path = "/tmp/glr_tinyc_table.csv";
        table_gen
            .export_to_csv_numeric(temp_path)
            .expect("Failed to export CSV");

        // Import and create parser
        let parser = RnglrParser::import_table_from_csv(temp_path).expect("Failed to import CSV");

        // Test with simple input: "{i=1;}"
        let test_input = "{i=1;}";

        // Tokenize
        let tokens = grammar.tokenize(test_input).expect("Failed to tokenize");

        // Convert to GLR format
        let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        // Parse
        let result = parser.parse(&glr_tokens);
        assert!(
            result.is_some(),
            "TinyC parser should accept '{}'",
            test_input
        );

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }
}
