//! Verification binary for the C regex lexer + Earley (Leo) parser.
//!
//! Pipeline:
//!   1. Compile the regex lexer spec  (c_regex.json)
//!   2. Load the tokenized grammar    (ansi_c_tok.json)
//!   3. Lex the C source file         → Vec<Token>
//!   4. Encode tokens to terminal IDs → Vec<u32>
//!   5. Parse with Earley/Leo         → ParseTree or reject
//!
//! Usage
//! -----
//!   cargo run --bin run_lexer -- <input_file> [grammar_json] [lexer_spec_json]
//!
//! Defaults
//! --------
//!   grammar_json    = grammars/ansi_c_tok.json
//!   lexer_spec_json = grammars/lexer/c_regex.json
//!
//! Example
//! -------
//!   cargo run --bin run_lexer -- input/ansi_c_small.txt

use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parsers::earley_leo;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: run_lexer <input_file> [grammar_json] [lexer_spec_json]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let grammar_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("grammars/ansi_c_tok.json");
    let lexer_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("grammars/lexer/c_regex.json");

    // ---- Load grammar --------------------------------------------------------
    let grammar = grammars::load_grammar_from_file(grammar_path).unwrap_or_else(|e| {
        eprintln!("Failed to load grammar from {:?}: {}", grammar_path, e);
        std::process::exit(1);
    });
    println!(
        "Grammar : {} ({} terminals, {} non-terminals)",
        grammar.name,
        grammar.num_terminals(),
        grammar.num_non_terminals()
    );

    // ---- Compile lexer -------------------------------------------------------
    let lexer = Lexer::from_file(lexer_path).unwrap_or_else(|e| {
        eprintln!("Failed to compile lexer from {:?}: {}", lexer_path, e);
        std::process::exit(1);
    });

    // ---- Read input ----------------------------------------------------------
    let input = fs::read_to_string(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {:?}: {}", input_path, e);
        std::process::exit(1);
    });
    println!("Input   : {} bytes from {:?}", input.len(), input_path);

    // ---- Lex -----------------------------------------------------------------
    let tokens = lexer.lex(&input).unwrap_or_else(|e| {
        eprintln!("Lex error: {}", e);
        std::process::exit(1);
    });
    println!("Tokens  : {}", tokens.len());

    // Print up to 40 tokens so large files stay readable
    let show = tokens.len().min(40);
    for tok in &tokens[..show] {
        println!(
            "  [{:>15}]  {:?}  @ byte {}",
            tok.kind, tok.text, tok.offset
        );
    }
    if tokens.len() > show {
        println!("  ... ({} more tokens)", tokens.len() - show);
    }

    // ---- Encode --------------------------------------------------------------
    let ids = encode(&tokens, &grammar).unwrap_or_else(|e| {
        eprintln!("\nEncoding error: {}", e);
        // Show every distinct unknown terminal to aid debugging
        let mut unknown: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for tok in &tokens {
            if grammar.terminals.get_id(&tok.kind).is_none() {
                unknown.insert(&tok.kind);
            }
        }
        eprintln!("Unknown terminals: {:?}", unknown);
        std::process::exit(1);
    });
    println!("\nEncoded : {} terminal IDs", ids.len());

    // ---- Parse (Earley + Leo) ------------------------------------------------
    println!("\n--- Earley/Leo parse ---");
    let mut parser = earley_leo::LeoParser::new(grammar);
    match parser.parse(ids) {
        Some(tree) => {
            println!("Result  : ACCEPT");
            // Display a portion of the tree (cap lines to avoid flooding stdout)
            let tree_str = tree.display();
            let lines: Vec<&str> = tree_str.lines().collect();
            let cap = 80;
            for line in lines.iter().take(cap) {
                println!("{}", line);
            }
            if lines.len() > cap {
                println!("  ... ({} more lines)", lines.len() - cap);
            }
        }
        None => {
            println!("Result  : REJECT");
            std::process::exit(2);
        }
    }
}
