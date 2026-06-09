use parser_comparison::grammars::{self, NumericGrammar};
use parser_comparison::lexer::{encode, Lexer, Token};
use parser_comparison::parsers::gll;
use std::env;
use std::fs;
use std::time::Instant;

const DEFAULT_GRAMMAR_PATH: &str = "grammars/ansi_c_tok.json";
const DEFAULT_LEXER_PATH: &str = "grammars/lexer/c_regex.json";
const DEFAULT_INPUT_PATH: &str = "test.cpp";
const TOKEN_PREVIEW_LIMIT: usize = 60;

struct Config {
    input_label: String,
    input: String,
    grammar_path: String,
    lexer_path: Option<String>,
}

fn main() {
    let config = match parse_args() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("{}", e);
            print_usage();
            return;
        }
    };

    let grammar = match grammars::load_grammar_from_file(&config.grammar_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading grammar {}: {}", config.grammar_path, e);
            return;
        }
    };

    println!(
        "Grammar: {} ({} terminals, {} non-terminals, {} productions)",
        grammar.name,
        grammar.num_terminals(),
        grammar.num_non_terminals(),
        grammar.production_count()
    );
    println!(
        "Input:   {} ({} bytes)",
        config.input_label,
        config.input.len()
    );

    let tokens = match tokenize(&grammar, &config.input, config.lexer_path.as_deref()) {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Tokenization failed: {}", e);
            return;
        }
    };

    println!("Tokens:  {} terminal IDs", tokens.len());

    let mut parser = gll::GLLParser::new(&grammar);
    let start = Instant::now();
    let result = parser.parse(&tokens);
    let elapsed = start.elapsed();

    match result {
        Some(_) => println!("\n[✓] Parse succeeded in {:.2?}", elapsed),
        None => println!("\n[✗] Parse failed in {:.2?}", elapsed),
    }
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Err(String::new());
    }

    let (input_label, input, rest) = if args.first().map(String::as_str) == Some("--text") {
        let text = args
            .get(1)
            .ok_or_else(|| "--text requires a source string".to_string())?
            .clone();
        ("<text>".to_string(), text, args[2..].to_vec())
    } else {
        let input_path = args
            .first()
            .cloned()
            .unwrap_or_else(|| DEFAULT_INPUT_PATH.to_string());
        let input = fs::read_to_string(&input_path)
            .map_err(|e| format!("Error reading input {}: {}", input_path, e))?;
        (input_path, input, args.get(1..).unwrap_or(&[]).to_vec())
    };

    let grammar_path = rest
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_GRAMMAR_PATH.to_string());

    let lexer_path = match rest.get(1).map(String::as_str) {
        Some("--char") | Some("char") | Some("none") => None,
        Some(path) => Some(path.to_string()),
        None => Some(DEFAULT_LEXER_PATH.to_string()),
    };

    Ok(Config {
        input_label,
        input,
        grammar_path,
        lexer_path,
    })
}

fn tokenize(
    grammar: &NumericGrammar,
    input: &str,
    lexer_path: Option<&str>,
) -> Result<Vec<u32>, String> {
    match lexer_path {
        Some(path) => {
            println!("Lexer:   {}", path);
            let lexer = Lexer::from_file(path)?;
            let lexed = lexer.lex(input)?;
            print_lexed_preview(&lexed);
            encode(&lexed, grammar)
        }
        None => {
            println!("Lexer:   character tokenizer");
            let tokens = grammar.tokenize(input).ok_or_else(|| {
                "input contains characters that are not terminals in the grammar".to_string()
            })?;
            print_terminal_preview(grammar, &tokens);
            Ok(tokens)
        }
    }
}

fn print_lexed_preview(tokens: &[Token]) {
    let show = tokens.len().min(TOKEN_PREVIEW_LIMIT);
    for tok in &tokens[..show] {
        println!("  [{:>12}] {:?} @ byte {}", tok.kind, tok.text, tok.offset);
    }
    if tokens.len() > show {
        println!("  ... ({} more tokens)", tokens.len() - show);
    }
}

fn print_terminal_preview(grammar: &NumericGrammar, tokens: &[u32]) {
    let show = tokens.len().min(TOKEN_PREVIEW_LIMIT);
    let names: Vec<&str> = tokens[..show]
        .iter()
        .map(|id| grammar.terminal_str(*id).unwrap_or("?"))
        .collect();
    println!("  {}", names.join(" "));
    if tokens.len() > show {
        println!("  ... ({} more tokens)", tokens.len() - show);
    }
}

fn print_usage() {
    eprintln!(
        "Usage:\n  cargo run -- [input_file] [grammar_json] [lexer_json|--char]\n  cargo run -- --text '<source>' [grammar_json] [lexer_json|--char]\n\nDefaults:\n  input_file  = {}\n  grammar     = {}\n  lexer       = {}\n\nPass --char as the third positional argument to use NumericGrammar::tokenize for character grammars.",
        DEFAULT_INPUT_PATH, DEFAULT_GRAMMAR_PATH, DEFAULT_LEXER_PATH
    );
}
