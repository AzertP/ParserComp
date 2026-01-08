//! Tests for the grammars module - CNF transformations and grammar loading

use super::*;

#[test]
fn test_replace_terminal_symbols() {
    // Create a simple grammar:
    // S -> a B
    // B -> b
    let json = r#"{
        "name": "test",
        "start": "<S>",
        "rules": {
            "<S>": [["a", "<B>"]],
            "<B>": [["b"]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");

    println!("=== Original Grammar ===");
    println!("Non-terminals: {:?}", grammar.non_terminals);
    println!("Terminals: {:?}", grammar.terminals);
    println!("Rules:");
    for (lhs, productions) in &grammar.rules {
        let lhs_str = grammar.non_terminals.get_str(*lhs).unwrap_or("?");
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    NumSymbol::Terminal(id) => {
                        format!("'{}'", grammar.terminals.get_str(*id).unwrap_or("?"))
                    }
                    NumSymbol::NonTerminal(id) => grammar
                        .non_terminals
                        .get_str(*id)
                        .unwrap_or("?")
                        .to_string(),
                })
                .collect();
            println!("  {} -> {}", lhs_str, prod_str.join(" "));
        }
    }

    let cnf = grammar.to_str_grammar().replace_terminal_symbols();

    println!("\n=== After replace_terminal_symbols ===");
    println!("Non-terminals: {:?}", cnf.non_terminals);
    println!("Terminals: {:?}", cnf.terminals);
    println!("Rules:");
    for (lhs, productions) in &cnf.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    // Expected output:
    // <S> -> <a> <B>    (terminals replaced with new non-terminals)
    // <B> -> <b>
    // <a> -> 'a'        (new rule for terminal 'a')
    // <b> -> 'b'        (new rule for terminal 'b')
}

#[test]
fn test_decompose_grammar() {
    // Create a grammar with a long production:
    // S -> a b c d (4 symbols - needs decomposition)
    // B -> x y     (2 symbols - no decomposition needed)
    let json = r#"{
        "name": "test_decompose",
        "start": "<S>",
        "rules": {
            "<S>": [["<A>", "<B>", "<C>", "<D>"]],
            "<A>": [["a"]],
            "<B>": [["b"]],
            "<C>": [["c"]],
            "<D>": [["d"]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
    let str_grammar = grammar.to_str_grammar();

    println!("=== Original Grammar ===");
    for (lhs, productions) in &str_grammar.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    let decomposed = str_grammar.decompose_grammar();

    // Verify start symbol is preserved
    assert_eq!(decomposed.start, "<S>", "Start symbol should be preserved");

    println!("\n=== After decompose_grammar ===");
    println!("Start: {}", decomposed.start);
    println!("Non-terminals: {:?}", decomposed.non_terminals);
    for (lhs, productions) in &decomposed.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    // Verify all productions have at most 2 symbols
    for (lhs, productions) in &decomposed.rules {
        for prod in productions {
            assert!(
                prod.len() <= 2,
                "Production {} -> {:?} has {} symbols (max 2 allowed)",
                lhs,
                prod,
                prod.len()
            );
        }
    }

    // Expected decomposition:
    // <S> -> <A> <S_0_>        (first two, rest moved to new rule)
    // <S_0_> -> <B> <S_0__>
    // <S_0__> -> <C> <D>
    // <A> -> 'a'
    // <B> -> 'b'
    // <C> -> 'c'
    // <D> -> 'd'
}

#[test]
fn test_eliminate_epsilon() {
    // Grammar with epsilon productions:
    // S -> A B
    // A -> a | ε
    // B -> b | ε
    // After elimination, we should get:
    // S -> A B | A | B
    // A -> a
    // B -> b
    let json = r#"{
        "name": "test_epsilon",
        "start": "<S>",
        "rules": {
            "<S>": [["<A>", "<B>"]],
            "<A>": [["a"], []],
            "<B>": [["b"], []]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
    let str_grammar = grammar.to_str_grammar();

    println!("=== Original Grammar ===");
    for (lhs, productions) in &str_grammar.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            if prod_str.is_empty() {
                println!("  {} -> ε", lhs);
            } else {
                println!("  {} -> {}", lhs, prod_str.join(" "));
            }
        }
    }

    let epsilon_free = str_grammar.eliminate_epsilon();

    println!("\n=== After eliminate_epsilon ===");
    for (lhs, productions) in &epsilon_free.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    // Verify no epsilon productions remain
    for (lhs, productions) in &epsilon_free.rules {
        for prod in productions {
            assert!(
                !prod.is_empty(),
                "Production {} -> ε should have been eliminated",
                lhs
            );
        }
    }

    // Verify S has the expected productions: A B, A, B
    let s_prods = epsilon_free
        .rules
        .get("<S>")
        .expect("<S> should have productions");
    assert!(
        s_prods.len() == 3,
        "S should have 3 productions, got {}",
        s_prods.len()
    );

    // Check that A -> a exists
    let a_prods = epsilon_free
        .rules
        .get("<A>")
        .expect("<A> should have productions");
    assert!(a_prods.len() == 1, "A should have 1 production");
    assert!(matches!(&a_prods[0][0], StrSymbol::Terminal(t) if t == "a"));

    // Check that B -> b exists
    let b_prods = epsilon_free
        .rules
        .get("<B>")
        .expect("<B> should have productions");
    assert!(b_prods.len() == 1, "B should have 1 production");
    assert!(matches!(&b_prods[0][0], StrSymbol::Terminal(t) if t == "b"));
}

#[test]
fn test_eliminate_epsilon_chain() {
    // Test transitively nullable non-terminals:
    // S -> A B
    // A -> C
    // B -> b
    // C -> ε
    // C is directly nullable, A is transitively nullable
    let json = r#"{
        "name": "test_epsilon_chain",
        "start": "<S>",
        "rules": {
            "<S>": [["<A>", "<B>"]],
            "<A>": [["<C>"]],
            "<B>": [["b"]],
            "<C>": [[]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
    let str_grammar = grammar.to_str_grammar();

    println!("=== Original Grammar (chain) ===");
    for (lhs, productions) in &str_grammar.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            if prod_str.is_empty() {
                println!("  {} -> ε", lhs);
            } else {
                println!("  {} -> {}", lhs, prod_str.join(" "));
            }
        }
    }

    let epsilon_free = str_grammar.eliminate_epsilon();

    println!("\n=== After eliminate_epsilon ===");
    for (lhs, productions) in &epsilon_free.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    // Verify no epsilon productions remain
    for (lhs, productions) in &epsilon_free.rules {
        for prod in productions {
            assert!(
                !prod.is_empty(),
                "Production {} -> ε should have been eliminated",
                lhs
            );
        }
    }

    // S should have productions for both: A B and B (since A is nullable)
    let s_prods = epsilon_free
        .rules
        .get("<S>")
        .expect("<S> should have productions");
    assert!(
        s_prods.len() == 2,
        "S should have 2 productions, got {}",
        s_prods.len()
    );
}

#[test]
fn test_remove_unit_rules() {
    // Grammar with unit rules:
    // S -> A
    // A -> B
    // B -> a | b c
    // After unit rule elimination:
    // S -> a | b c
    // A -> a | b c
    // B -> a | b c
    let json = r#"{
        "name": "test_unit",
        "start": "<S>",
        "rules": {
            "<S>": [["<A>"]],
            "<A>": [["<B>"]],
            "<B>": [["a"], ["b", "c"]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
    let str_grammar = grammar.to_str_grammar();

    println!("=== Original Grammar ===");
    for (lhs, productions) in &str_grammar.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    let unit_free = str_grammar.remove_unit_rules();

    println!("\n=== After remove_unit_rules ===");
    for (lhs, productions) in &unit_free.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    // Verify no unit rules remain
    for (lhs, productions) in &unit_free.rules {
        for prod in productions {
            let is_unit = prod.len() == 1 && matches!(&prod[0], StrSymbol::NonTerminal(_));
            assert!(
                !is_unit,
                "Unit production {} -> {:?} should have been eliminated",
                lhs, prod
            );
        }
    }

    // S should now have the same productions as B (transitively)
    let s_prods = unit_free
        .rules
        .get("<S>")
        .expect("<S> should have productions");
    assert!(
        s_prods.len() == 2,
        "S should have 2 productions, got {}",
        s_prods.len()
    );

    // A should also have the same productions as B
    let a_prods = unit_free
        .rules
        .get("<A>")
        .expect("<A> should have productions");
    assert!(
        a_prods.len() == 2,
        "A should have 2 productions, got {}",
        a_prods.len()
    );
}

#[test]
fn test_remove_unit_rules_cycle() {
    // Grammar with a cycle of unit rules:
    // S -> A | x
    // A -> B
    // B -> A | y
    // The cycle A -> B -> A should be handled correctly
    let json = r#"{
        "name": "test_unit_cycle",
        "start": "<S>",
        "rules": {
            "<S>": [["<A>"], ["x"]],
            "<A>": [["<B>"]],
            "<B>": [["<A>"], ["y"]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
    let str_grammar = grammar.to_str_grammar();

    println!("=== Original Grammar (cycle) ===");
    for (lhs, productions) in &str_grammar.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    let unit_free = str_grammar.remove_unit_rules();

    println!("\n=== After remove_unit_rules ===");
    for (lhs, productions) in &unit_free.rules {
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    StrSymbol::Terminal(t) => format!("'{}'", t),
                    StrSymbol::NonTerminal(nt) => nt.clone(),
                })
                .collect();
            println!("  {} -> {}", lhs, prod_str.join(" "));
        }
    }

    // Verify no unit rules remain
    for (lhs, productions) in &unit_free.rules {
        for prod in productions {
            let is_unit = prod.len() == 1 && matches!(&prod[0], StrSymbol::NonTerminal(_));
            assert!(
                !is_unit,
                "Unit production {} -> {:?} should have been eliminated",
                lhs, prod
            );
        }
    }

    // S should have x and y (from the transitive closure through the cycle)
    let s_prods = unit_free
        .rules
        .get("<S>")
        .expect("<S> should have productions");
    assert!(
        s_prods.len() == 2,
        "S should have 2 productions (x and y), got {}",
        s_prods.len()
    );
}

#[test]
fn test_to_cnf() {
    // Test the complete CNF conversion pipeline
    // Grammar:
    // S -> A B C
    // A -> a | ε
    // B -> b
    // C -> c
    //
    // After CNF conversion, all productions should be either:
    // 1. A -> a (single terminal)
    // 2. A -> B C (exactly two non-terminals)
    let json = r#"{
        "name": "test_cnf",
        "start": "<S>",
        "rules": {
            "<S>": [["<A>", "<B>", "<C>"]],
            "<A>": [["a"], []],
            "<B>": [["b"]],
            "<C>": [["c"]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");

    println!("=== Original Grammar ===");
    println!("Start: {:?}", grammar.start_str());
    for (nt_id, productions) in &grammar.rules {
        let nt_name = grammar.non_terminal_str(*nt_id).unwrap_or("?");
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    NumSymbol::Terminal(id) => {
                        format!("'{}'", grammar.terminal_str(*id).unwrap_or("?"))
                    }
                    NumSymbol::NonTerminal(id) => {
                        grammar.non_terminal_str(*id).unwrap_or("?").to_string()
                    }
                })
                .collect();
            if prod_str.is_empty() {
                println!("  {} -> ε", nt_name);
            } else {
                println!("  {} -> {}", nt_name, prod_str.join(" "));
            }
        }
    }

    let cnf = grammar.to_cnf();

    println!("\n=== After to_cnf ===");
    println!("Start: {:?}", cnf.start_str());
    for (nt_id, productions) in &cnf.rules {
        let nt_name = cnf.non_terminal_str(*nt_id).unwrap_or("?");
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    NumSymbol::Terminal(id) => {
                        format!("'{}'", cnf.terminal_str(*id).unwrap_or("?"))
                    }
                    NumSymbol::NonTerminal(id) => {
                        cnf.non_terminal_str(*id).unwrap_or("?").to_string()
                    }
                })
                .collect();
            println!("  {} -> {}", nt_name, prod_str.join(" "));
        }
    }

    // Verify CNF properties
    for (nt_id, productions) in &cnf.rules {
        let nt_name = cnf.non_terminal_str(*nt_id).unwrap_or("?");
        for prod in productions {
            // No empty productions
            assert!(
                !prod.is_empty(),
                "CNF should not have empty productions: {} -> ε",
                nt_name
            );

            if prod.len() == 1 {
                // Single symbol must be a terminal
                assert!(
                    prod[0].is_terminal(),
                    "Single-symbol production must be terminal: {} -> {:?}",
                    nt_name,
                    prod
                );
            } else if prod.len() == 2 {
                // Two symbols must both be non-terminals
                assert!(
                    prod[0].is_non_terminal() && prod[1].is_non_terminal(),
                    "Two-symbol production must be two non-terminals: {} -> {:?}",
                    nt_name,
                    prod
                );
            } else {
                panic!(
                    "CNF production must have 1 or 2 symbols: {} -> {:?} (has {})",
                    nt_name,
                    prod,
                    prod.len()
                );
            }
        }
    }

    println!("\nCNF validation passed!");
}

#[test]
fn test_to_cnf_complex() {
    // More complex grammar test
    // S -> a b c d (long production with terminals mixed)
    let json = r#"{
        "name": "test_cnf_complex",
        "start": "<S>",
        "rules": {
            "<S>": [["a", "b", "c", "d"]]
        }
    }"#;

    let grammar = load_grammar_from_str(json).expect("Failed to load grammar");
    let cnf = grammar.to_cnf();

    println!("=== Complex CNF ===");
    for (nt_id, productions) in &cnf.rules {
        let nt_name = cnf.non_terminal_str(*nt_id).unwrap_or("?");
        for prod in productions {
            let prod_str: Vec<String> = prod
                .iter()
                .map(|s| match s {
                    NumSymbol::Terminal(id) => {
                        format!("'{}'", cnf.terminal_str(*id).unwrap_or("?"))
                    }
                    NumSymbol::NonTerminal(id) => {
                        cnf.non_terminal_str(*id).unwrap_or("?").to_string()
                    }
                })
                .collect();
            println!("  {} -> {}", nt_name, prod_str.join(" "));
        }
    }

    // Verify CNF properties
    for (nt_id, productions) in &cnf.rules {
        let nt_name = cnf.non_terminal_str(*nt_id).unwrap_or("?");
        for prod in productions {
            assert!(!prod.is_empty(), "No empty productions allowed");

            if prod.len() == 1 {
                assert!(
                    prod[0].is_terminal(),
                    "Single must be terminal: {} -> {:?}",
                    nt_name,
                    prod
                );
            } else {
                assert!(
                    prod.len() == 2,
                    "Must have exactly 2 symbols: {} -> {:?}",
                    nt_name,
                    prod
                );
                assert!(
                    prod[0].is_non_terminal() && prod[1].is_non_terminal(),
                    "Both must be non-terminals: {} -> {:?}",
                    nt_name,
                    prod
                );
            }
        }
    }

    println!("\nComplex CNF validation passed!");
}
