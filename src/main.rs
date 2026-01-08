use parser_comparison::grammars::{self, Grammar};
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{cyk, earley, gll, glr, valiant};
use std::fs;
use std::time::Instant;

// Configuration
// Change these paths to test different grammars and inputs, and LR tables

const GRAMMAR_PATH: &str = "grammars/css.json";
const INPUT_PATH: &str = "input/css_small.txt";
const TABLE_PATH: &str = "table/css_glr_table.csv";

/// Result of a parser benchmark run
#[allow(dead_code)]
struct BenchmarkResult {
    parser_name: String,
    success: bool,
    duration_us: u128,
}

/// Run a single parser and measure its execution time
#[allow(dead_code)]
fn benchmark_parser<F>(name: &str, parse_fn: F) -> BenchmarkResult
where
    F: FnOnce() -> Option<ParseTree>,
{
    let start: Instant = Instant::now();
    let result = parse_fn();
    let duration = start.elapsed();

    BenchmarkResult {
        parser_name: name.to_string(),
        success: result.is_some(),
        duration_us: duration.as_millis(),
    }
}

/// Run all parsers on the given grammar and numeric input
fn run_comparison(
    grammar: &Grammar,
    cnf_grammar: &Grammar,
    input: &str,
    rnglr: &glr::RnglrParser,
    brnglr: &glr::BrnglrParser,
    gll_parser: &mut gll::GLLParser,
    earley: &mut earley::EarleyParser
) {
    // Tokenize input
    let tokens = grammar.tokenize(input).expect("Failed to tokenize input");
    let cnf_tokens = cnf_grammar
        .tokenize(input)
        .expect("Failed to tokenize input");

    // Convert tokens to i32 for GLR parser (with +1 offset for terminals)
    // The GLR table format uses: 0 = end-of-input, terminals = ID + 1, non-terminals = -(ID + 1)
    let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

    let results: Vec<BenchmarkResult> = vec![
        // benchmark_parser("CYK", || cyk::parse(cnf_grammar, &cnf_tokens)),
        // benchmark_parser("Valiant", || valiant::parse(cnf_grammar, &cnf_tokens)),
        benchmark_parser("Earley", || earley.parse(tokens.clone())),
        benchmark_parser("GLL", || gll_parser.parse(&tokens)),
        benchmark_parser("RNGLR", || rnglr.parse(&glr_tokens)),
        benchmark_parser("BRNGLR", || brnglr.parse(&glr_tokens)),
    ];

    // Print results
    for res in results {
        if res.success == false {
            eprintln!("Parser {} failed to parse the input.", res.parser_name);
        }
        println!(
            "[{}] {} in {}ms",
            if res.success { "✓" } else { "✗" },
            res.parser_name,
            res.duration_us
        );
    }
}

/// Load input from file
fn load_input(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("Failed to read input file '{}': {}", path, e))
}

/// Parse and run all test inputs from the file
/// Each non-empty line is treated as a separate test input
fn run_all_tests(grammar: &Grammar, input_content: &str) {
    let lines: Vec<&str> = input_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    println!("\nFound {} test inputs in file", lines.len());
    println!("{}", "=".repeat(60));

    let cnf_grammar = grammar.to_cnf();

    // Generate GLR parsing table
    let table_generator = table_generator::TableGenerator::new(grammar);
    table_generator
        .export_to_csv_numeric(TABLE_PATH)
        .expect("Failed to export GLR table to CSV");

    let rnglr_parser = glr::RnglrParser::import_table_from_csv(TABLE_PATH)
        .expect("Failed to load RNGLR table from CSV");
    let brnglr_parser = glr::BrnglrParser::import_table_from_csv(TABLE_PATH)
        .expect("Failed to load BRNGLR table from CSV");
    let mut gll_parser = gll::GLLParser::new(grammar);
    let mut earley_parser = earley::EarleyParser::new(grammar.clone());

    // Limit to first N lines for quicker testing
    let lines_to_run = &lines[..std::cmp::min(10, lines.len())];

    for (i, line) in lines_to_run.iter().enumerate() {
        // Print test number, length of input and snippet
        println!(
            "\nTest #{}: {} ({} bytes)",
            i + 1,
            line[..std::cmp::min(50, line.len())].trim(),
            line.len()
        );

        run_comparison(
            grammar,
            &cnf_grammar,
            *line,
            &rnglr_parser,
            &brnglr_parser,
            &mut gll_parser,
            &mut earley_parser,
        );
    }

    println!("\n{}", "=".repeat(60));
}

fn main() {
    println!("Parser Comparison Framework (Rust)");
    println!("===================================\n");
    println!("Grammar: {}", GRAMMAR_PATH);
    println!("Input:   {}\n", INPUT_PATH);

    // Load the grammar
    let grammar = match grammars::load_grammar_from_file(GRAMMAR_PATH) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading grammar: {}", e);
            eprintln!("Make sure to run from the project root directory.");
            return;
        }
    };

    println!("Loaded grammar: {}", grammar.name);
    println!(
        "  Start symbol: {} (ID: {})",
        grammar.start_str().unwrap_or("?"),
        grammar.start
    );
    println!("  Non-terminals: {}", grammar.num_non_terminals());
    println!("  Terminals: {}", grammar.num_terminals());
    println!("  Productions: {}", grammar.production_count());

    // Load the input file
    let input_content = match load_input(INPUT_PATH) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error loading input: {}", e);
            return;
        }
    };

    println!("\nInput file loaded: {} bytes", input_content.len());

    // Run tests - each line in the input file is a separate test
    run_all_tests(&grammar, &input_content);

    println!("\nComparison complete!");
}
