use parser_comparison::grammars;
use parser_comparison::parsers::earley;
use std::fs;
use std::time::Instant;

const GRAMMAR_PATH: &str = "grammars/cpp_ws.json";
const INPUT_PATH: &str = "test.cpp";

fn main() {
    // Load grammar
    let grammar = match grammars::load_grammar_from_file(GRAMMAR_PATH) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading grammar: {}", e);
            return;
        }
    };

    println!("Grammar: {} ({} rules)", grammar.name, grammar.production_count());

    // Load test.java
    let input = match fs::read_to_string(INPUT_PATH) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", INPUT_PATH, e);
            return;
        }
    };

    println!("Input:   {} ({} bytes)", INPUT_PATH, input.len());

    // Tokenize
    let tokens = match grammar.tokenize(&input) {
        Some(t) => t,
        None => {
            eprintln!("Tokenization failed: input contains characters not in the grammar.");
            return;
        }
    };

    println!("Tokens:  {} tokens", tokens.len());

    // Parse with Earley
    let mut parser = earley::EarleyParser::new(grammar);
    let start = Instant::now();
    let result = parser.parse(tokens);
    let elapsed = start.elapsed();

    match result {
        Some(_) => println!("\n[✓] Parse succeeded in {:.2?}", elapsed),
        None    => println!("\n[✗] Parse failed in {:.2?}", elapsed),
    }
}
