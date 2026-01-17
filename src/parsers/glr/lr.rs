
use crate::grammars::{Grammar, NumSymbol};
use crate::parsers::glr::table_generator::{TableGenerator, Action, END_OF_INPUT};
use crate::parse_tree::ParseTree;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// Use faster hash map
type FastHashMap<K, V> = FxHashMap<K, V>;

pub struct LRParser {
    // State -> Symbol -> Action
    table: FastHashMap<usize, FastHashMap<NumSymbol, Action>>,
    grammar: Grammar,
}

impl LRParser {
    pub fn new(grammar: &Grammar) -> Self {
        let generator = TableGenerator::new(grammar);
        let table_raw = generator.generate_lr1_table();
        
        // Convert to faster hash map and flattened action
        let mut table = FastHashMap::default();
        for (state, actions) in table_raw {
            let mut state_table = FastHashMap::default();
            for (symbol, acts) in actions {
                if acts.len() > 1 {
                    panic!("Conflict detected in LR(1) table at state {}, symbol {:?}: {:?}", state, symbol, acts);
                }
                if !acts.is_empty() {
                    state_table.insert(symbol, acts[0].clone());
                }
            }
            table.insert(state, state_table);
        }

        LRParser {
            table,
            grammar: grammar.clone(),
        }
    }

    /// Load LR(1) parse table from a CSV file (numeric format)
    pub fn from_csv<P: AsRef<Path>>(path: P, grammar: &Grammar) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut lines = reader.lines();

        // 1. Read header to identify column symbols
        // Format: state,sym1,sym2,...
        let header_line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty CSV file"))??;
        let columns: Vec<&str> = header_line.split(',').collect();

        // Skip first column "state" and parse symbols
        let mut column_symbols = Vec::new();
        for col in columns.iter().skip(1) {
            let val = col.trim().parse::<i64>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid symbol in header: {}", e),
                )
            })?;
            
            let symbol = if val == 0 {
                NumSymbol::Terminal(END_OF_INPUT)
            } else if val > 0 {
                // Terminal: id + 1
                NumSymbol::Terminal((val - 1) as u32)
            } else {
                // NonTerminal: -(id + 1)
                NumSymbol::NonTerminal((-val - 1) as u32)
            };
            column_symbols.push(symbol);
        }

        let mut table = FastHashMap::default();

        // 2. Read data rows
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() != columns.len() {
                // Handle potential mismatch or empty lines
                continue;
            }

            let state_id = parts[0].parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid state ID: {}", e),
                )
            })?;
            
            let mut state_actions = FastHashMap::default();

            for (i, part) in parts.iter().skip(1).enumerate() {
                let action_str = part.trim();
                
                if action_str.is_empty() {
                    continue;
                }
                
                // Check for conflicts in CSV ( multiple actions separated by '/' )
                if action_str.contains('/') {
                    panic!("Conflict detected in LR(1) CSV table at state {}, column {}: {}", state_id, i, action_str);
                }

                let first_action_str = action_str;
                if first_action_str.is_empty() {
                    continue;
                }

                let action = if first_action_str.starts_with("p.") {
                    // Shift: p.state
                    if let Ok(next_state) = first_action_str[2..].parse::<usize>() {
                        Action::Shift(next_state)
                    } else {
                        continue;
                    }
                } else if first_action_str.starts_with("r.") {
                    // Reduce: r.{lhs_encoded}.{dot}.{label}
                    // lhs_encoded is -(lhs_id + 1)
                    let comps: Vec<&str> = first_action_str.split('.').collect();
                    if comps.len() >= 4 {
                        let lhs_encoded = comps[1].parse::<i64>().unwrap_or(0);
                        let lhs = (-lhs_encoded - 1) as u32;
                        let dot = comps[2].parse::<usize>().unwrap_or(0);
                        let label = comps[3].parse::<usize>().unwrap_or(0);
                        Action::Reduce(lhs, dot, label)
                    } else {
                        continue;
                    }
                } else if first_action_str == "acc" {
                    Action::Accept
                } else {
                    continue;
                };

                state_actions.insert(column_symbols[i], action);
            }
            table.insert(state_id, state_actions);
        }

        Ok(LRParser {
            table,
            grammar: grammar.clone(),
        })
    }

    /// Perform recognition on the input
    /// Input is a sequence of terminal IDs (as i32)
    /// - term_id + 1 for terminals
    /// - 0 for EOF (optional, can be implicit)
    pub fn recognize(&self, input: &[i32]) -> bool {
        self.parse(input).is_some()
    }

    /// Perform parsing on the input and return a ParseTree
    pub fn parse(&self, input: &[i32]) -> Option<ParseTree> {
        let mut stack = vec![0]; // Start state is always 0
        let mut node_stack: Vec<ParseTree> = Vec::new();
        let mut input_idx = 0;
        
        loop {
            // Peek at current state
            let current_state = match stack.last() {
                Some(&s) => s,
                None => return None, // Empty stack error
            };
            
            // Get lookahead symbol and raw value for tree construction
            let (lookahead, current_val) = if input_idx < input.len() {
                let val = input[input_idx];
                if val == 0 {
                    (NumSymbol::Terminal(END_OF_INPUT), 0)
                } else {
                    (NumSymbol::Terminal((val - 1) as u32), val)
                }
            } else {
                (NumSymbol::Terminal(END_OF_INPUT), 0)
            };

            // Get action from table
            let action = match self.table.get(&current_state).and_then(|t| t.get(&lookahead)) {
                Some(a) => a,
                None => return None, // No action defined -> Parse Error
            };

            match action {
                Action::Shift(next_state) => {
                    stack.push(*next_state);
                    
                    // Create tree node for terminal (unless it is EOF)
                    if lookahead != NumSymbol::Terminal(END_OF_INPUT) {
                         let term_id = (current_val - 1) as usize;
                         // Get terminal string from grammar or default
                         let name = self.grammar.terminals.get_str(term_id as u32)
                             .map(|s| s.to_string())
                             .unwrap_or_else(|| format!("t{}", term_id));
                         
                         // Terminals are leaves
                         node_stack.push(ParseTree::from_str(&name, vec![]));
                    }
                    
                    input_idx += 1; // Consume input
                }
                Action::Reduce(lhs, dot, _) => {
                    // Pop |rhs| items from state stack
                    for _ in 0..*dot {
                        if stack.pop().is_none() {
                            return None;
                        }
                    }
                    
                    // Pop |rhs| items from node stack to be children
                    let start = node_stack.len().checked_sub(*dot)?;
                    let children: Vec<ParseTree> = node_stack.drain(start..).collect();

                    // Create parent node
                    let lhs_name = self.grammar.non_terminals.get_str(*lhs)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("<{}>", lhs));
                    
                    let new_node = ParseTree::from_str(&lhs_name, children);

                    // Look at state after popping
                    let after_pop_state = match stack.last() {
                        Some(&s) => s,
                        None => return None,
                    };
                    
                    // Perform GOTO
                    let lhs_sym = NumSymbol::NonTerminal(*lhs);
                    match self.table.get(&after_pop_state).and_then(|t| t.get(&lhs_sym)) {
                        Some(Action::Shift(goto_state)) => {
                            stack.push(*goto_state);
                            node_stack.push(new_node);
                        }
                        _ => return None, // Missing GOTO -> Error
                    }
                    // Do NOT consume input
                }
                Action::Accept => {
                    // On accept, return the root of the tree
                    return node_stack.pop();
                }
            }
        }
    }
}















// ---------------------- TESTS ----------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars;
    use std::fs;

    #[test]
    fn test_lr_simple_grammar() {
        // Grammar: S -> a B c, B -> b
        let json = r#"{
            "name": "simple_lr",
            "start": "<S>",
            "rules": {
                "<S>": [["a", "<B>", "c"]],
                "<B>": [["b"]]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let table_gen = TableGenerator::new(&grammar);

        // Ensure lr_table directory exists
        let table_dir = "lr_table";
        fs::create_dir_all(table_dir).ok();

        // Export to CSV
        let table_path = format!("{}/simple_lr_table.csv", table_dir);
        table_gen
            .export_lr1_to_csv(&table_path)
            .expect("Failed to export CSV");
        println!("Exported LR parse table to {}", table_path);

        // Import and test parser
        let parser = LRParser::from_csv(&table_path, &grammar)
            .expect("Failed to import CSV into parser");
        
        // Test "abc"
        let input = "abc";
        let tokens = grammar.tokenize(input).expect("Failed to tokenize input");
        let lr_tokens: Vec<_> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        assert!(parser.recognize(&lr_tokens), "Should accept 'abc'");

        // Test invalid "ac"
        let tokens_invalid = grammar.tokenize("a").unwrap(); // partial
        let lr_tokens_invalid: Vec<_> = tokens_invalid.iter().map(|&t| (t + 1) as i32).collect();
        assert!(!parser.recognize(&lr_tokens_invalid), "Should reject 'a'");
    }

    #[test]
    fn test_lr_calc_grammar() {
         // Simple arithmetic: E -> E + T | T, T -> T * F | F, F -> ( E ) | n
         // Simplified to be LR(1) compatible (unambiguous, left-recursive is fine for LR)
         // E -> E + T
         // E -> T
         // T -> n
         let json = r#"{
            "name": "calc_simple",
            "start": "<E>",
            "rules": {
                "<E>": [["<E>", "+", "<T>"], ["<T>"]],
                "<T>": [["n"]]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let table_gen = TableGenerator::new(&grammar);

        let table_dir = "lr_table";
        fs::create_dir_all(table_dir).ok();
        let table_path = format!("{}/calc_lr_table.csv", table_dir);
        
        table_gen
            .export_lr1_to_csv(&table_path)
            .expect("Failed to export CSV");

        let parser = LRParser::from_csv(&table_path, &grammar)
            .expect("Failed to import CSV");

        // "n + n"
        let input = "n+n";
        let tokens = grammar.tokenize(input).expect("Failed to tokenize");
        let lr_tokens: Vec<_> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        assert!(parser.recognize(&lr_tokens), "Should accept 'n+n'");
        
        // "n + n + n"
        let input2 = "n+n+n";
        let tokens2 = grammar.tokenize(input2).expect("Failed");
        let lr_tokens2: Vec<_> = tokens2.iter().map(|&t| (t + 1) as i32).collect();
        assert!(parser.recognize(&lr_tokens2), "Should accept 'n+n+n'");
    }

    #[test]
    fn test_lr_parse_tree_display() {
        let json = r#"{
            "name": "simple_tree",
            "start": "<S>",
            "rules": {
                "<S>": [["a", "<B>", "c"]],
                "<B>": [["b"]]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let table_gen = TableGenerator::new(&grammar);

        let table_dir = "lr_table";
        fs::create_dir_all(table_dir).ok();
        let table_path = format!("{}/simple_tree_table.csv", table_dir);
        
        table_gen
            .export_lr1_to_csv(&table_path)
            .expect("Failed to export CSV");

        let parser = LRParser::from_csv(&table_path, &grammar)
            .expect("Failed to import CSV");

        let input = "abc";
        let tokens = grammar.tokenize(input).expect("Failed");
        let lr_tokens: Vec<_> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        match parser.parse(&lr_tokens) {
            Some(tree) => {
                println!("Parse tree for 'abc':\n{}", tree.display());
            }
            None => panic!("Failed to parse 'abc'"),
        }
    }

    #[test]
    fn test_lr_calc_tree_display() {
         let json = r#"{
            "name": "calc_tree",
            "start": "<E>",
            "rules": {
                "<E>": [["<E>", "+", "<T>"], ["<T>"]],
                "<T>": [["n"]]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let table_gen = TableGenerator::new(&grammar);

        let table_dir = "lr_table";
        fs::create_dir_all(table_dir).ok();
        let table_path = format!("{}/calc_tree_table.csv", table_dir);
        
        table_gen
            .export_lr1_to_csv(&table_path)
            .expect("Failed to export CSV");

        let parser = LRParser::from_csv(&table_path, &grammar)
            .expect("Failed to import CSV");

        let input = "n+n+n";
        let tokens = grammar.tokenize(input).expect("Failed");
        let lr_tokens: Vec<_> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        match parser.parse(&lr_tokens) {
            Some(tree) => {
                println!("Parse tree for 'n+n+n':\n{}", tree.display());
            }
            None => panic!("Failed to parse 'n+n+n'"),
        }
    }
}
