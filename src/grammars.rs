// Grammars module - loads grammars from JSON files and converts to numerical representation

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

// ============================================================================
// Symbol Table - maps between strings and numeric IDs
// ============================================================================

/// Bidirectional mapping between symbols (strings) and numeric IDs
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// String to ID mapping
    str_to_id: HashMap<String, u32>,
    /// ID to String mapping
    id_to_str: Vec<String>,
}

impl SymbolTable {
    /// Create a new empty symbol table
    pub fn new() -> Self {
        SymbolTable {
            str_to_id: HashMap::new(),
            id_to_str: Vec::new(),
        }
    }

    /// Get or create an ID for a symbol string
    pub fn get_or_insert(&mut self, symbol: &str) -> u32 {
        if let Some(&id) = self.str_to_id.get(symbol) {
            id
        } else {
            let id = self.id_to_str.len() as u32;
            self.str_to_id.insert(symbol.to_string(), id);
            self.id_to_str.push(symbol.to_string());
            id
        }
    }

    /// Get ID for a symbol (returns None if not found)
    pub fn get_id(&self, symbol: &str) -> Option<u32> {
        self.str_to_id.get(symbol).copied()
    }

    /// Get string for an ID (returns None if not found)
    pub fn get_str(&self, id: u32) -> Option<&str> {
        self.id_to_str.get(id as usize).map(|s| s.as_str())
    }

    /// Get the number of symbols in the table
    pub fn len(&self) -> usize {
        self.id_to_str.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_str.is_empty()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Numeric Grammar Representation
// ============================================================================

/// A numeric symbol - either a terminal or non-terminal identified by ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumSymbol {
    Terminal(u32),
    NonTerminal(u32),
}

impl NumSymbol {
    /// Check if this is a terminal
    pub fn is_terminal(&self) -> bool {
        matches!(self, NumSymbol::Terminal(_))
    }

    /// Check if this is a non-terminal
    pub fn is_non_terminal(&self) -> bool {
        matches!(self, NumSymbol::NonTerminal(_))
    }

    /// Get the numeric ID
    pub fn id(&self) -> u32 {
        match self {
            NumSymbol::Terminal(id) | NumSymbol::NonTerminal(id) => *id,
        }
    }
}

/// A numeric production is a list of numeric symbols
pub type NumProduction = Vec<NumSymbol>;

/// Numeric grammar representation - all symbols are numeric IDs
#[derive(Debug, Clone)]
pub struct NumericGrammar {
    pub name: String,
    pub start: u32,
    pub rules: HashMap<u32, Vec<NumProduction>>,
    pub tests: Vec<Vec<u32>>,
    pub terminals: SymbolTable,
    pub non_terminals: SymbolTable,
}

impl NumericGrammar {
    /// Create a new empty numeric grammar
    pub fn new(name: &str) -> Self {
        NumericGrammar {
            name: name.to_string(),
            start: 0,
            rules: HashMap::new(),
            tests: Vec::new(),
            terminals: SymbolTable::new(),
            non_terminals: SymbolTable::new(),
        }
    }

    /// Get the string representation of a terminal ID
    pub fn terminal_str(&self, id: u32) -> Option<&str> {
        self.terminals.get_str(id)
    }

    /// Get the string representation of a non-terminal ID
    pub fn non_terminal_str(&self, id: u32) -> Option<&str> {
        self.non_terminals.get_str(id)
    }

    /// Convert a numeric symbol back to its string representation
    pub fn symbol_to_str(&self, sym: &NumSymbol) -> Option<&str> {
        match sym {
            NumSymbol::Terminal(id) => self.terminals.get_str(*id),
            NumSymbol::NonTerminal(id) => self.non_terminals.get_str(*id),
        }
    }

    /// Convert a test input back to string
    pub fn test_to_str(&self, test: &[u32]) -> String {
        test.iter()
            .filter_map(|&id| self.terminals.get_str(id))
            .collect()
    }

    /// Tokenize an input string to numeric terminal IDs
    /// Returns None if any character is not in the terminal table
    pub fn tokenize(&self, input: &str) -> Option<Vec<u32>> {
        input
            .chars()
            .map(|c| self.terminals.get_id(&c.to_string()))
            .collect()
    }

    /// Get the number of terminals
    pub fn num_terminals(&self) -> usize {
        self.terminals.len()
    }

    /// Get the number of non-terminals
    pub fn num_non_terminals(&self) -> usize {
        self.non_terminals.len()
    }

    /// Count total number of productions
    pub fn production_count(&self) -> usize {
        self.rules.values().map(|v| v.len()).sum()
    }

    /// Get productions for a non-terminal
    pub fn get_productions(&self, nt: u32) -> Option<&Vec<NumProduction>> {
        self.rules.get(&nt)
    }

    /// Get the start symbol's string name
    pub fn start_str(&self) -> Option<&str> {
        self.non_terminals.get_str(self.start)
    }

    /// Convert numeric grammar to string-based grammar
    fn to_str_grammar(&self) -> StrGrammar {
        let mut rules: HashMap<String, Vec<Vec<StrSymbol>>> = HashMap::new();
        let mut terminals: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut non_terminals: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (lhs_id, productions) in &self.rules {
            let lhs_str = self.non_terminals.get_str(*lhs_id).unwrap().to_string();
            non_terminals.insert(lhs_str.clone());

            let str_productions: Vec<Vec<StrSymbol>> = productions
                .iter()
                .map(|prod| {
                    prod.iter()
                        .map(|sym| match sym {
                            NumSymbol::Terminal(id) => {
                                let term_str = self.terminals.get_str(*id).unwrap().to_string();
                                terminals.insert(term_str.clone());
                                StrSymbol::Terminal(term_str)
                            }
                            NumSymbol::NonTerminal(id) => {
                                let nt_str = self.non_terminals.get_str(*id).unwrap().to_string();
                                non_terminals.insert(nt_str.clone());
                                StrSymbol::NonTerminal(nt_str)
                            }
                        })
                        .collect()
                })
                .collect();
            rules.insert(lhs_str, str_productions);
        }

        let start = self.non_terminals.get_str(self.start).unwrap().to_string();
        let tests: Vec<String> = self.tests.iter().map(|t| self.test_to_str(t)).collect();

        StrGrammar {
            name: self.name.clone(),
            start,
            rules,
            tests,
            terminals,
            non_terminals,
        }
    }

    pub fn debug_print(&self) {
        println!("=== Grammar: {} ===", self.name);
        println!(
            "Start symbol: {} (id={})",
            self.start_str().unwrap_or("?"),
            self.start
        );
        println!(
            "Terminals: {:?}",
            (0..self.terminals.len())
                .map(|i| self.terminals.get_str(i as u32).unwrap())
                .collect::<Vec<_>>()
        );
        println!(
            "Non-terminals: {:?}",
            (0..self.non_terminals.len())
                .map(|i| self.non_terminals.get_str(i as u32).unwrap())
                .collect::<Vec<_>>()
        );
        println!("\nRules:");

        // Sort keys for deterministic output
        let mut sorted_keys: Vec<_> = self.rules.keys().copied().collect();
        sorted_keys.sort();

        for nt_id in sorted_keys {
            let nt_str = self.non_terminal_str(nt_id).unwrap_or("?");
            let productions = &self.rules[&nt_id];

            for prod in productions {
                let rhs: Vec<String> = prod
                    .iter()
                    .map(|sym| match sym {
                        NumSymbol::Terminal(id) => {
                            format!("'{}'", self.terminal_str(*id).unwrap_or("?"))
                        }
                        NumSymbol::NonTerminal(id) => {
                            self.non_terminal_str(*id).unwrap_or("?").to_string()
                        }
                    })
                    .collect();

                if rhs.is_empty() {
                    println!("  {} -> ε", nt_str);
                } else {
                    println!("  {} -> {}", nt_str, rhs.join(" "));
                }
            }
        }
        println!("=== End Grammar ===\n");
    }
}

// ============================================================================
// String-based Grammar for CNF transformations
// ============================================================================

/// String-based grammar for easier CNF manipulation
#[derive(Debug, Clone)]
struct StrGrammar {
    name: String,
    start: String,
    rules: HashMap<String, Vec<Vec<StrSymbol>>>,
    tests: Vec<String>,
    /// Set of terminal symbols (strings)
    terminals: std::collections::HashSet<String>,
    /// Set of non-terminal symbols (strings)
    non_terminals: std::collections::HashSet<String>,
}

#[allow(dead_code)]
impl StrGrammar {
    /// Create a new StrGrammar with explicit terminal/non-terminal sets
    fn new(
        name: String,
        start: String,
        rules: HashMap<String, Vec<Vec<StrSymbol>>>,
        tests: Vec<String>,
        terminals: std::collections::HashSet<String>,
        non_terminals: std::collections::HashSet<String>,
    ) -> Self {
        StrGrammar {
            name,
            start,
            rules,
            tests,
            terminals,
            non_terminals,
        }
    }

    /// Check if a symbol string is a terminal
    fn is_terminal(&self, s: &str) -> bool {
        self.terminals.contains(s)
    }

    /// Check if a symbol string is a non-terminal
    fn is_non_terminal(&self, s: &str) -> bool {
        self.non_terminals.contains(s)
    }

    /// Convert a string to the appropriate StrSymbol based on known sets
    fn to_symbol(&self, s: &str) -> StrSymbol {
        if self.terminals.contains(s) {
            StrSymbol::Terminal(s.to_string())
        } else {
            StrSymbol::NonTerminal(s.to_string())
        }
    }

    /// Add a new terminal to the grammar
    fn add_terminal(&mut self, s: &str) {
        self.terminals.insert(s.to_string());
    }

    /// Add a new non-terminal to the grammar
    fn add_non_terminal(&mut self, s: &str) {
        self.non_terminals.insert(s.to_string());
    }

    /// Convert string grammar back to numeric grammar
    fn to_numeric_grammar(&self) -> NumericGrammar {
        let mut grammar = NumericGrammar::new(&self.name);

        // Register start symbol first (ID 0)
        grammar.non_terminals.get_or_insert(&self.start);

        // Register all non-terminals
        for lhs in self.rules.keys() {
            grammar.non_terminals.get_or_insert(lhs);
        }

        // Register all symbols from productions
        for productions in self.rules.values() {
            for prod in productions {
                for sym in prod {
                    match sym {
                        StrSymbol::Terminal(s) => {
                            grammar.terminals.get_or_insert(s);
                        }
                        StrSymbol::NonTerminal(s) => {
                            grammar.non_terminals.get_or_insert(s);
                        }
                    }
                }
            }
        }

        // Set start symbol
        grammar.start = grammar.non_terminals.get_id(&self.start).unwrap();

        // Convert rules
        for (lhs, productions) in &self.rules {
            let lhs_id = grammar.non_terminals.get_id(lhs).unwrap();
            let num_productions: Vec<NumProduction> = productions
                .iter()
                .map(|prod| {
                    prod.iter()
                        .map(|sym| match sym {
                            StrSymbol::Terminal(s) => {
                                NumSymbol::Terminal(grammar.terminals.get_id(s).unwrap())
                            }
                            StrSymbol::NonTerminal(s) => {
                                NumSymbol::NonTerminal(grammar.non_terminals.get_id(s).unwrap())
                            }
                        })
                        .collect()
                })
                .collect();
            grammar.rules.insert(lhs_id, num_productions);
        }

        // Convert tests
        for test in &self.tests {
            let tokens: Vec<u32> = test
                .chars()
                .map(|c| grammar.terminals.get_or_insert(&c.to_string()))
                .collect();
            grammar.tests.push(tokens);
        }

        grammar
    }

    // Replace terminal symbols in productions with new non-terminals
    fn replace_terminal_symbols(&self) -> StrGrammar {
        let mut new_rules: HashMap<String, Vec<Vec<StrSymbol>>> = HashMap::new();
        let new_terminals = self.terminals.clone();
        let mut new_non_terminals = self.non_terminals.clone();

        for (lhs, productions) in &self.rules {
            let mut new_productions = Vec::new();

            for rhs in productions {
                // Only replace terminals if production has more than one symbol
                if rhs.len() <= 1 {
                    new_productions.push(rhs.clone());
                    continue;
                }

                let mut new_rhs = Vec::new();
                for symbol in rhs {
                    match symbol {
                        StrSymbol::Terminal(term) => {
                            // Create new non-terminal name for this terminal
                            let new_nt = format!("<{}>", term);
                            new_non_terminals.insert(new_nt.clone());
                            // Add rule: <a> -> 'a' (if not already added)
                            new_rules
                                .entry(new_nt.clone())
                                .or_default()
                                .push(vec![StrSymbol::Terminal(term.clone())]);
                            // Replace terminal with new non-terminal
                            new_rhs.push(StrSymbol::NonTerminal(new_nt));
                        }
                        StrSymbol::NonTerminal(_) => {
                            new_rhs.push(symbol.clone());
                        }
                    }
                }
                new_productions.push(new_rhs);
            }
            new_rules
                .entry(lhs.clone())
                .or_default()
                .extend(new_productions);
        }

        // Deduplicate productions for terminal rules (e.g., multiple <a> -> 'a')
        for productions in new_rules.values_mut() {
            productions.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
            productions.dedup();
        }

        StrGrammar {
            name: self.name.clone(),
            start: self.start.clone(),
            rules: new_rules,
            tests: self.tests.clone(),
            terminals: new_terminals,
            non_terminals: new_non_terminals,
        }
    }

    // Next, we want to replace any rule that contains more than two tokens with
    // it decomposition.
    // [] = []
    // [t1] = [t1]
    // [t1, t2] = [t1, t2]
    // [t1, t2, t3] = [t1, _t2], _t2 = [t2, t3]
    fn decompose_rule(
        rule: &[StrSymbol],
        prefix: &str,
    ) -> (Vec<StrSymbol>, HashMap<String, Vec<Vec<StrSymbol>>>) {
        let l = rule.len();
        if l <= 2 {
            return (rule.to_vec(), HashMap::new());
        }

        let t = rule[0].clone();
        let r = &rule[1..];

        let kp = format!("{}_", prefix);
        let (nr, mut d) = Self::decompose_rule(r, &kp);

        let k = format!("<{}>", kp);
        d.insert(k.clone(), vec![nr]);

        (vec![t, StrSymbol::NonTerminal(k)], d)
    }

    fn decompose_grammar(&self) -> StrGrammar {
        // Implementation of decomposition step
        let mut new_rules: HashMap<String, Vec<Vec<StrSymbol>>> = HashMap::new();
        let mut new_non_terminals = self.non_terminals.clone();

        for (k, productions) in &self.rules {
            let mut new_productions = Vec::new();
            for (i, r) in productions.iter().enumerate() {
                // Extract the inner part of the key (remove < and >)
                let inner = k.trim_start_matches('<').trim_end_matches('>');
                let prefix = format!("{}_{}", inner, i);

                let (nr, d) = Self::decompose_rule(r, &prefix);
                new_productions.push(nr);

                for (new_k, new_v) in d {
                    new_non_terminals.insert(new_k.clone());
                    new_rules.entry(new_k).or_default().extend(new_v);
                }
            }

            new_rules
                .entry(k.clone())
                .or_default()
                .extend(new_productions);
        }

        StrGrammar {
            name: self.name.clone(),
            start: self.start.clone(),
            rules: new_rules,
            tests: self.tests.clone(),
            terminals: self.terminals.clone(),
            non_terminals: new_non_terminals,
        }
    }

    fn eliminate_epsilon(&self) -> StrGrammar {
        // Implementation of epsilon elimination step
        let mut nullable_set = HashSet::new();
        let mut changed = true;

        // First, find all nullable non-terminals
        while changed {
            changed = false;
            for lhs in self.rules.keys() {
                if nullable_set.contains(lhs) {
                    continue; // Already known to be nullable
                }
                for rhs in &self.rules[lhs] {
                    // Epsilon rule nt -> []
                    if rhs.is_empty() {
                        if nullable_set.insert(lhs.clone()) {
                            changed = true;
                        }
                        break; // Found nullable
                    }
                    // If all symbols on RHS are nullable non-terminals
                    let all_nullable = rhs.iter().all(|sym| {
                        match sym {
                            StrSymbol::NonTerminal(nt) => nullable_set.contains(nt),
                            StrSymbol::Terminal(_) => false, // Terminals are never nullable
                        }
                    });
                    if all_nullable {
                        if nullable_set.insert(lhs.clone()) {
                            changed = true;
                        }
                        break; // Found nullable
                    }
                }
            }
        }

        // Now, create new grammar without epsilon productions
        let mut new_rules = HashMap::new();
        for (lhs, productions) in &self.rules {
            let mut new_productions = Vec::new();
            for rhs in productions {
                // Epsione rule, ignore
                if rhs.is_empty() {
                    continue;
                }

                // Terminal rules, keep as is
                if rhs.len() == 1 {
                    new_productions.push(rhs.clone());
                    continue;
                }

                // For length = 2
                if rhs.len() == 2 {
                    let (a, b) = (&rhs[0], &rhs[1]);
                    let null_a = match a {
                        StrSymbol::NonTerminal(nt) => nullable_set.contains(nt),
                        StrSymbol::Terminal(_) => false,
                    };
                    let null_b = match b {
                        StrSymbol::NonTerminal(nt) => nullable_set.contains(nt),
                        StrSymbol::Terminal(_) => false,
                    };

                    new_productions.push(rhs.clone()); // Original rule
                    if null_a {
                        new_productions.push(vec![b.clone()]);
                    }
                    if null_b {
                        new_productions.push(vec![a.clone()]);
                    }
                }

                // For length > 2
                if rhs.len() > 2 {
                    println!("Warning: Production {} -> {:?} has length > 2", lhs, rhs);
                }
            }
            new_rules.insert(lhs.clone(), new_productions);
        }

        // Check empty RHS
        let empty_nts: HashSet<String> = self
            .rules
            .keys()
            .filter(|nt| !new_rules.contains_key(*nt))
            .cloned()
            .collect();

        for productions in new_rules.values_mut() {
            for prod in productions.iter_mut() {
                prod.retain(|sym| match sym {
                    StrSymbol::NonTerminal(nt) => !empty_nts.contains(nt),
                    StrSymbol::Terminal(_) => true,
                });
            }
        }

        // Remove duplicates and empty productions
        let mut final_rules: HashMap<String, Vec<Vec<StrSymbol>>> = HashMap::new();
        for (nt, productions) in new_rules {
            let mut seen: HashSet<Vec<StrSymbol>> = HashSet::new();
            let mut unique_productions = Vec::new();

            for prod in productions {
                if !prod.is_empty() && seen.insert(prod.clone()) {
                    unique_productions.push(prod);
                }
            }

            if !unique_productions.is_empty() {
                final_rules.insert(nt, unique_productions);
            }
        }

        // Update non_terminals set
        let mut new_non_terminals = HashSet::new();
        for nt in final_rules.keys() {
            new_non_terminals.insert(nt.clone());
        }
        for productions in final_rules.values() {
            for prod in productions {
                for sym in prod {
                    if let StrSymbol::NonTerminal(nt) = sym {
                        new_non_terminals.insert(nt.clone());
                    }
                }
            }
        }

        StrGrammar {
            name: self.name.clone(),
            start: self.start.clone(),
            rules: final_rules,
            tests: self.tests.clone(),
            terminals: self.terminals.clone(),
            non_terminals: new_non_terminals,
        }
    }

    fn remove_unit_rules(&self) -> StrGrammar {
        let mut new_rules = HashMap::new();

        for lhs in self.rules.keys() {
            let mut visited = HashSet::new();
            let mut queue = vec![lhs.clone()];
            let mut new_productions = Vec::new();

            while !queue.is_empty() {
                let current = queue.pop().unwrap();
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());

                if let Some(productions) = self.rules.get(&current) {
                    for production in productions {
                        if production.len() == 1
                            && matches!(production[0], StrSymbol::NonTerminal(_))
                        {
                            queue.push(match &production[0] {
                                StrSymbol::NonTerminal(nt) => nt.clone(),
                                _ => unreachable!(),
                            });
                        } else {
                            new_productions.push(production.clone());
                        }
                    }
                }
            }

            // Remove duplicates
            let mut seen: HashSet<Vec<StrSymbol>> = HashSet::new();
            new_productions.retain(|prod| seen.insert(prod.clone()));

            new_rules.insert(lhs.clone(), new_productions);
        }

        StrGrammar {
            name: self.name.clone(),
            start: self.start.clone(),
            rules: new_rules,
            tests: self.tests.clone(),
            terminals: self.terminals.clone(),
            non_terminals: self.non_terminals.clone(),
        }
    }
}

// Converting grammars to CNF
impl NumericGrammar {
    /// Convert the grammar to Chomsky Normal Form (CNF)
    pub fn to_cnf(&self) -> NumericGrammar {
        let mut cnf_grammar = self.to_str_grammar();
        cnf_grammar = cnf_grammar.replace_terminal_symbols();
        cnf_grammar = cnf_grammar.decompose_grammar();
        cnf_grammar = cnf_grammar.eliminate_epsilon();
        cnf_grammar = cnf_grammar.remove_unit_rules();
        // g3 = balance_grammar(g2)
        // g3['<>'] = [[]]
        // g4 = eliminate_epsilon(g3)
        // g5 = remove_unit_rules(g4)
        cnf_grammar.to_numeric_grammar()
    }
}

// ============================================================================
// String-based Grammar (intermediate representation for JSON loading)
// ============================================================================

/// String-based symbol for parsing JSON
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StrSymbol {
    Terminal(String),
    NonTerminal(String),
}

/// JSON structure for grammar files
#[derive(Debug, Deserialize)]
struct GrammarJson {
    name: String,
    start: String,
    rules: HashMap<String, Value>,
    #[serde(default)]
    tests: Vec<String>,
}

// ============================================================================
// Grammar Loading and Conversion
// ============================================================================

/// Load a grammar from a JSON file and convert to numeric representation
pub fn load_grammar_from_file<P: AsRef<Path>>(path: P) -> Result<NumericGrammar, String> {
    let content = fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?;
    load_grammar_from_str(&content)
}

/// Load a grammar from a JSON string and convert to numeric representation
pub fn load_grammar_from_str(json: &str) -> Result<NumericGrammar, String> {
    let parsed: GrammarJson =
        serde_json::from_str(json).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let mut grammar = NumericGrammar::new(&parsed.name);

    // First pass: collect all symbols and build tables
    let mut str_rules: HashMap<String, Vec<Vec<StrSymbol>>> = HashMap::new();

    for (lhs, rhs_value) in &parsed.rules {
        let productions = parse_rules(rhs_value)?;
        str_rules.insert(lhs.clone(), productions);
    }

    // Register non-terminals (start symbol first to ensure it gets ID 0)
    grammar.non_terminals.get_or_insert(&parsed.start);
    
    // Sort non-terminal names for deterministic ordering
    let mut nt_names: Vec<String> = str_rules.keys().cloned().collect();
    nt_names.sort();
    
    for lhs in &nt_names {
        grammar.non_terminals.get_or_insert(lhs);
    }

    // Register all terminals and non-terminals from productions
    // Sort to ensure deterministic ordering
    for lhs in &nt_names {
        let productions = &str_rules[lhs];
        for prod in productions {
            for sym in prod {
                match sym {
                    StrSymbol::Terminal(s) => {
                        grammar.terminals.get_or_insert(s);
                    }
                    StrSymbol::NonTerminal(s) => {
                        grammar.non_terminals.get_or_insert(s);
                    }
                }
            }
        }
    }

    // Set start symbol
    grammar.start = grammar
        .non_terminals
        .get_id(&parsed.start)
        .ok_or_else(|| format!("Start symbol '{}' not found", parsed.start))?;

    // Convert rules to numeric form
    for (lhs, productions) in str_rules {
        let lhs_id = grammar.non_terminals.get_id(&lhs).unwrap();
        let num_productions: Vec<NumProduction> = productions
            .into_iter()
            .map(|prod| {
                prod.into_iter()
                    .map(|sym| match sym {
                        StrSymbol::Terminal(s) => {
                            NumSymbol::Terminal(grammar.terminals.get_id(&s).unwrap())
                        }
                        StrSymbol::NonTerminal(s) => {
                            NumSymbol::NonTerminal(grammar.non_terminals.get_id(&s).unwrap())
                        }
                    })
                    .collect()
            })
            .collect();
        grammar.rules.insert(lhs_id, num_productions);
    }

    // Convert test strings to numeric form
    for test in &parsed.tests {
        let tokens: Vec<u32> = test
            .chars()
            .map(|c| grammar.terminals.get_or_insert(&c.to_string()))
            .collect();
        grammar.tests.push(tokens);
    }

    Ok(grammar)
}

/// Parse rules from JSON value
fn parse_rules(value: &Value) -> Result<Vec<Vec<StrSymbol>>, String> {
    match value {
        // Array of productions: [["a", "<B>"], ["c"]]
        Value::Array(productions) => {
            let mut result = Vec::new();
            for prod in productions {
                match prod {
                    Value::Array(symbols) => {
                        let production = parse_production(symbols)?;
                        result.push(production);
                    }
                    _ => return Err("Production must be an array".to_string()),
                }
            }
            Ok(result)
        }
        // Special rule object: {"digits": true} or {"letters": true} or {"char_range": [...]}
        Value::Object(obj) => {
            let mut result = Vec::new();

            if obj.get("digits").and_then(|v| v.as_bool()) == Some(true) {
                for i in 0..=9 {
                    result.push(vec![StrSymbol::Terminal(i.to_string())]);
                }
            } else if obj.get("letters").and_then(|v| v.as_bool()) == Some(true) {
                for c in 'a'..='z' {
                    result.push(vec![StrSymbol::Terminal(c.to_string())]);
                }
            } else if let Some(range) = obj.get("char_range") {
                let range_arr = range.as_array().ok_or("char_range must be an array")?;
                let start = range_arr
                    .get(0)
                    .and_then(|v| v.as_u64())
                    .ok_or("char_range start must be a number")? as u8;
                let end = range_arr
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .ok_or("char_range end must be a number")? as u8;

                let exclude: Vec<u8> = obj
                    .get("exclude")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u8))
                            .collect()
                    })
                    .unwrap_or_default();

                for i in start..end {
                    if !exclude.contains(&i) {
                        result.push(vec![StrSymbol::Terminal((i as char).to_string())]);
                    }
                }
            }

            Ok(result)
        }
        _ => Err("Rules must be an array or object".to_string()),
    }
}

/// Parse a single production from JSON array
fn parse_production(symbols: &[Value]) -> Result<Vec<StrSymbol>, String> {
    symbols
        .iter()
        .map(|s| {
            let sym_str = s.as_str().ok_or("Symbol must be a string")?;
            if sym_str.starts_with('<') && sym_str.ends_with('>') {
                Ok(StrSymbol::NonTerminal(sym_str.to_string()))
            } else {
                Ok(StrSymbol::Terminal(sym_str.to_string()))
            }
        })
        .collect()
}

// ============================================================================
// Grammar loading functions
// ============================================================================

/// Load all grammars from the grammars directory
pub fn load_all_grammars(grammars_dir: &str) -> Result<Vec<NumericGrammar>, String> {
    let mut grammars = Vec::new();
    let dir = Path::new(grammars_dir);

    if !dir.exists() {
        return Err(format!("Grammars directory not found: {}", grammars_dir));
    }

    let entries = fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            match load_grammar_from_file(&path) {
                Ok(grammar) => grammars.push(grammar),
                Err(e) => eprintln!("Warning: Failed to load {:?}: {}", path, e),
            }
        }
    }

    Ok(grammars)
}

/// Load a specific grammar by name from the grammars directory
pub fn load_grammar(grammars_dir: &str, name: &str) -> Result<NumericGrammar, String> {
    let path = Path::new(grammars_dir).join(format!("{}.json", name));
    load_grammar_from_file(&path)
}

// ============================================================================
// Convenience functions to load built-in grammars
// ============================================================================

const GRAMMARS_DIR: &str = "grammars";

pub fn simple_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "simple").expect("Failed to load simple grammar")
}

pub fn ambi_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "ambi").expect("Failed to load ambi grammar")
}

pub fn calc_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "calc").expect("Failed to load calc grammar")
}

pub fn json_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "json").expect("Failed to load json grammar")
}

pub fn tinyc_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "tinyc").expect("Failed to load tinyc grammar")
}

pub fn sexp_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "sexp").expect("Failed to load sexp grammar")
}

pub fn newline_grammar() -> NumericGrammar {
    load_grammar(GRAMMARS_DIR, "newline").expect("Failed to load newline grammar")
}

// ============================================================================
// Re-export types for backward compatibility
// ============================================================================

pub type Grammar = NumericGrammar;
pub type Symbol = NumSymbol;
pub type Production = NumProduction;

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "grammars_tests.rs"]
mod tests;
