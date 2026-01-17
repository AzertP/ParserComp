use crate::grammars::{NumSymbol, NumericGrammar};
use crate::parse_tree::ParseTree;
use std::collections::{HashMap, HashSet};

pub struct LLParser {
    grammar: NumericGrammar,
    first: HashMap<u32, HashSet<u32>>,
    follow: HashMap<u32, HashSet<u32>>,
    nullable: HashSet<u32>,
    /// Parse table: (NonTerminal ID, Lookahead Terminal ID) -> Production Index
    /// None for Lookahead means EOF
    table: HashMap<(u32, Option<u32>), usize>,
}

impl LLParser {
    pub fn new(grammar: &NumericGrammar) -> Self {
        let (first, follow, nullable) = Self::compute_first_follow_nullable(grammar);
        let mut parser = LLParser {
            grammar: grammar.clone(),
            first,
            follow,
            nullable,
            table: HashMap::new(),
        };
        parser.compute_parse_table();
        parser
    }

    /// Returns true if the input is accepted by the grammar
    pub fn recognize(&self, input: &[u32]) -> bool {
        let mut stack: Vec<NumSymbol> = vec![NumSymbol::NonTerminal(self.grammar.start)];
        let mut input_idx = 0;

        while let Some(top) = stack.pop() {
            let current_token = if input_idx < input.len() {
                Some(input[input_idx])
            } else {
                None
            };

            match top {
                NumSymbol::Terminal(t) => {
                    if let Some(token) = current_token {
                        if t == token {
                            input_idx += 1;
                        } else {
                            return false; // Mismatch
                        }
                    } else {
                        return false; // Unexpected EOF
                    }
                }
                NumSymbol::NonTerminal(nt) => {
                    if let Some(&prod_idx) = self.table.get(&(nt, current_token)) {
                        let production = &self.grammar.rules[&nt][prod_idx];
                        // Push symbols in reverse order
                        for sym in production.iter().rev() {
                            stack.push(*sym);
                        }
                    } else {
                        return false; // No rule for (NT, token)
                    }
                }
            }
        }

        // Accepted if stack is empty (loop condition) AND input is fully consumed
        input_idx == input.len()
    }

    /// Parse the input and return a ParseTree
    pub fn parse(&self, input: &[u32]) -> Option<ParseTree> {
        let mut cursor = 0;
        let start_symbol = NumSymbol::NonTerminal(self.grammar.start);

        let root = self.parse_recursive(start_symbol, input, &mut cursor)?;

        // Ensure all input is consumed
        if cursor == input.len() {
            Some(root)
        } else {
            None
        }
    }

    fn parse_recursive(
        &self,
        symbol: NumSymbol,
        input: &[u32],
        cursor: &mut usize,
    ) -> Option<ParseTree> {
        match symbol {
            NumSymbol::Terminal(t) => {
                let current_token = input.get(*cursor);
                if let Some(&token) = current_token {
                    if t == token {
                        *cursor += 1;
                        let token_str = self.grammar.terminal_str(t).unwrap_or("?");
                        return Some(ParseTree::leaf(token_str));
                    }
                }
                None
            }
            NumSymbol::NonTerminal(nt) => {
                // Lookahead
                let current_token = input.get(*cursor).copied();

                if let Some(&prod_idx) = self.table.get(&(nt, current_token)) {
                    let production = &self.grammar.rules[&nt][prod_idx];
                    let nt_str = self.grammar.non_terminal_str(nt).unwrap_or("?");

                    let mut children = Vec::new();
                    for &sym in production {
                        if let Some(child) = self.parse_recursive(sym, input, cursor) {
                            children.push(child);
                        } else {
                            return None;
                        }
                    }
                    Some(ParseTree::from_str(nt_str, children))
                } else {
                    None
                }
            }
        }
    }

    pub fn parse_bool(&self, input: &[i32]) -> bool {
        // Compatibility wrapper for original signature
        // Assuming i32 input fits in u32 token IDs or is compatible
        let u32_input: Vec<u32> = input.iter().map(|&x| x as u32).collect();
        self.recognize(&u32_input)
    }

    fn compute_parse_table(&mut self) {
        for (&nt, productions) in &self.grammar.rules {
            for (prod_idx, production) in productions.iter().enumerate() {
                let (first_alpha, alpha_nullable) = self.first_of_sequence(production);

                // Rule 1: For each a in FIRST(alpha), add A -> alpha to M[A, a]
                for &t in &first_alpha {
                    if let Some(existing) = self.table.insert((nt, Some(t)), prod_idx) {
                        if existing != prod_idx {
                            panic!("LL(1) Conflict at NonTerminal {} (ID: {}), Token {} (ID: {}). Rules {} and {}", 
                                   self.grammar.non_terminal_str(nt).unwrap_or("?"), nt, 
                                   self.grammar.terminal_str(t).unwrap_or("?"), t,
                                   existing, prod_idx);
                        }
                    }
                }

                // Rule 2: If alpha is nullable, for each b in FOLLOW(A), add A -> alpha to M[A, b]
                if alpha_nullable {
                    if let Some(follow_set) = self.follow.get(&nt) {
                        for &b in follow_set {
                            if let Some(existing) = self.table.insert((nt, Some(b)), prod_idx) {
                                if existing != prod_idx {
                                    panic!("LL(1) Conflict (Follow) at NonTerminal {} (ID: {}), Token {} (ID: {}). Rules {} and {}", 
                                           self.grammar.non_terminal_str(nt).unwrap_or("?"), nt, 
                                           self.grammar.terminal_str(b).unwrap_or("?"), b,
                                           existing, prod_idx);
                                }
                            }
                        }
                    }
                    // TODO: Handle EOF in FOLLOW if explicit EOF is not in the set logic,
                    // but typically simple grammars might not strictly model EOF token in FOLLOW.
                    // If we assume EOF is valid if nullable and we run out of input:
                    // We can add an entry for None (EOF)
                    // Checking if the start symbol or generally if FOLLOW contains EOF-equivalent.
                    // For now, let's assume if nullable, we add entry for EOF (None) if it's potentially valid.
                    // A simple heuristic: if nullable, add EOF entry.
                    if let Some(existing) = self.table.insert((nt, None), prod_idx) {
                        if existing != prod_idx {
                            panic!(
                                "LL(1) Conflict (EOF) at NonTerminal {} (ID: {}). Rules {} and {}",
                                self.grammar.non_terminal_str(nt).unwrap_or("?"),
                                nt,
                                existing,
                                prod_idx
                            );
                        }
                    }
                }
            }
        }
    }

    /// Compute FIRST set of a sequence of symbols
    fn first_of_sequence(&self, sequence: &[NumSymbol]) -> (HashSet<u32>, bool) {
        let mut result_first = HashSet::new();
        let mut is_seq_nullable = true;

        for sym in sequence {
            match sym {
                NumSymbol::Terminal(t) => {
                    result_first.insert(*t);
                    is_seq_nullable = false;
                    break;
                }
                NumSymbol::NonTerminal(nt) => {
                    if let Some(nt_first) = self.first.get(nt) {
                        result_first.extend(nt_first);
                    }
                    if !self.nullable.contains(nt) {
                        is_seq_nullable = false;
                        break;
                    }
                }
            }
        }

        (result_first, is_seq_nullable)
    }

    /// Compute FIRST, FOLLOW, and NULLABLE sets

    fn compute_first_follow_nullable(
        grammar: &NumericGrammar,
    ) -> (
        HashMap<u32, HashSet<u32>>,
        HashMap<u32, HashSet<u32>>,
        HashSet<u32>,
    ) {
        let mut first: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut follow: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut nullable: HashSet<u32> = HashSet::new();

        // Initialize FIRST sets
        for &nt in grammar.rules.keys() {
            first.insert(nt, HashSet::new());
            follow.insert(nt, HashSet::new());
        }

        // Fixed-point iteration
        loop {
            let mut changed = false;

            for (&nt, rules) in &grammar.rules {
                for rule in rules {
                    // Check if rule is nullable
                    let mut can_be_empty = true;
                    for sym in rule {
                        match sym {
                            NumSymbol::Terminal(t) => {
                                // Add terminal to FIRST
                                if first.entry(nt).or_default().insert(*t) {
                                    changed = true;
                                }
                                can_be_empty = false;
                                break;
                            }
                            NumSymbol::NonTerminal(nt2) => {
                                // Add FIRST(nt2) to FIRST(nt)
                                let first_nt2 = first.get(nt2).cloned().unwrap_or_default();
                                for t in first_nt2 {
                                    if first.entry(nt).or_default().insert(t) {
                                        changed = true;
                                    }
                                }
                                if !nullable.contains(nt2) {
                                    can_be_empty = false;
                                    break;
                                }
                            }
                        }
                    }
                    if can_be_empty {
                        if nullable.insert(nt) {
                            changed = true;
                        }
                    }

                    // Compute FOLLOW
                    let mut follow_set = follow.get(&nt).cloned().unwrap_or_default();
                    for sym in rule.iter().rev() {
                        match sym {
                            NumSymbol::Terminal(t) => {
                                follow_set = HashSet::new();
                                follow_set.insert(*t);
                            }
                            NumSymbol::NonTerminal(nt2) => {
                                let follow_nt2 = follow.entry(*nt2).or_default();
                                for t in &follow_set {
                                    if follow_nt2.insert(*t) {
                                        changed = true;
                                    }
                                }
                                if nullable.contains(nt2) {
                                    let first_nt2 = first.get(nt2).cloned().unwrap_or_default();
                                    follow_set = follow_set.union(&first_nt2).cloned().collect();
                                } else {
                                    follow_set = first.get(nt2).cloned().unwrap_or_default();
                                }
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        (first, follow, nullable)
    }
}

#[cfg(test)]
#[path = "ll_tests.rs"]
mod tests;
