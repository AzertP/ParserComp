//! Benchmarking tool that outputs CSV data for plotting parsing time vs input length.
//!
//! Usage:
//!   cargo run --release --bin benchmark_csv
//!
//! Output:
//!   Creates a CSV file in results/ with columns:
//!   parser, input_length, tokens, median_time_ns, mad_ns, iterations
use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::earley;
use parser_comparison::parsers::gll::ll;
use parser_comparison::parsers::glr::lr;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{cyk, earley_leo, gll, glr, valiant};
use parser_comparison::tree;
use std::collections::HashSet;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Grammar configurations to benchmark
struct GrammarConfig {
    name: &'static str,
    grammar_path: &'static str,
    input_paths: &'static [&'static str],
    table_path: &'static str,
    lr_table_path: &'static str,
    generate_table: bool,
    parsers: &'static [&'static str],
}

const ALL_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR", "CYK", "LL", "LR"];

const CONFIGS: &[GrammarConfig] = &[
    // ----- ANSI C -----
    GrammarConfig {
        name: "ansi_c",
        grammar_path: "grammars/ansi_c.json",
        input_paths: &["input/ansi_c_small.txt", "input/ansi_c_large.txt"],
        table_path: "table/ansi_c_glr_table.csv",
        lr_table_path: "table/ansi_c_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- Bool -----
    GrammarConfig {
        name: "bool",
        grammar_path: "grammars/bool.json",
        input_paths: &["input/bool.txt"],
        table_path: "table/bool_glr_table.csv",
        lr_table_path: "table/bool_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- Calculator -----
    GrammarConfig {
        name: "calc",
        grammar_path: "grammars/calc.json",
        input_paths: &["input/calc_small.txt", "input/calc_large.txt"],
        table_path: "table/calc_glr_table.csv",
        lr_table_path: "table/calc_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- Calculator LL(1) -----
    GrammarConfig {
        name: "calc_ll1",
        grammar_path: "grammars/ll1_calc.json",
        input_paths: &["input/calc_small.txt", "input/calc_large.txt"],
        table_path: "table/calc_ll1_glr_table.csv",
        lr_table_path: "table/calc_ll1_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- Expr -----
    GrammarConfig {
        name: "expr",
        grammar_path: "grammars/expr.json",
        input_paths: &["input/expr.txt"],
        table_path: "table/expr_glr_table.csv",
        lr_table_path: "table/expr_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- Expr Ambiguous -----
    GrammarConfig {
        name: "expr_ambi",
        grammar_path: "grammars/expr_ambi.json",
        input_paths: &["input/expr_ambi.txt"],
        table_path: "table/expr_ambi_glr_table.csv",
        lr_table_path: "table/expr_ambi_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- C++ -----
    GrammarConfig {
        name: "cpp",
        grammar_path: "grammars/cpp.json",
        input_paths: &["input/cpp_small.txt", "input/cpp_large.txt"],
        table_path: "table/cpp_glr_table.csv",
        lr_table_path: "table/cpp_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- CSS -----
    GrammarConfig {
        name: "css",
        grammar_path: "grammars/css.json",
        input_paths: &["input/css_small.txt"],
        table_path: "table/css_glr_table.csv",
        lr_table_path: "table/css_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- HTML -----
    GrammarConfig {
        name: "html",
        grammar_path: "grammars/html.json",
        input_paths: &["input/html.txt"],
        table_path: "table/html_glr_table.csv",
        lr_table_path: "table/html_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- Json -----
    GrammarConfig {
        name: "json",
        grammar_path: "grammars/json.json",
        input_paths: &["input/json_small.txt", "input/json_medium.txt", "input/json_large.txt", "input/json_ultra.txt"],
        table_path: "table/json_glr_table.csv",
        lr_table_path: "table/json_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- Json LL(1) -----
    GrammarConfig {
        name: "json_ll1",
        grammar_path: "grammars/ll1_json.json",
        input_paths: &["input/json_small.txt", "input/json_medium.txt", "input/json_large.txt", "input/json_ultra.txt"],
        table_path: "table/json_ll1_glr_table.csv",
        lr_table_path: "table/json_ll1_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- Json LR(1) -----
    GrammarConfig {
        name: "json_lr",
        grammar_path: "grammars/lr_json.json",
        input_paths: &["input/json_small.txt", "input/json_medium.txt", "input/json_large.txt", "input/json_ultra.txt"],
        table_path: "table/lr_json_glr_table.csv",
        lr_table_path: "table/lr_json_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- Json Ambiguous -----
    GrammarConfig {
        name: "json_ambi",
        grammar_path: "grammars/json_ambi.json",
        input_paths: &["input/json_small.txt", "input/json_medium.txt", "input/json_large.txt", "input/json_ultra.txt"],
        table_path: "table/json_ambi_glr_table.csv",
        lr_table_path: "table/json_ambi_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- Java -----
    GrammarConfig {
        name: "java",
        grammar_path: "grammars/jsl18.json",
        input_paths: &["input/java_small.txt", "input/java_large.txt"],
        table_path: "table/java_glr_table.csv",
        lr_table_path: "table/java_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- Pascal -----
    GrammarConfig {
        name: "pascal",
        grammar_path: "grammars/pascal.json",
        input_paths: &["input/pascal_small.txt", "input/pascal_large.txt"],
        table_path: "table/pascal_glr_table.csv",
        lr_table_path: "table/pascal_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- TinyPascal -----
    GrammarConfig {
        name: "tinypascal",
        grammar_path: "grammars/ll1_tinypascal.json",
        input_paths: &["input/tinypascal_large.txt"],
        table_path: "table/tinypascal_glr_table.csv",
        lr_table_path: "table/tinypascal_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- S-exp -----
    GrammarConfig {
        name: "sexp",
        grammar_path: "grammars/sexp.json",
        input_paths: &["input/sexp_small.txt", "input/sexp_large.txt"],
        table_path: "table/sexp_glr_table.csv",
        lr_table_path: "table/sexp_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- S-exp LL(1) -----
    GrammarConfig {
        name: "sexp_ll1",
        grammar_path: "grammars/ll1_sexp.json",
        input_paths: &["input/sexp_ll1.txt"],
        table_path: "table/sexp_ll1_glr_table.csv",
        lr_table_path: "table/sexp_ll1_lr_table.csv",
        generate_table: true,
        parsers: ALL_PARSERS,
    },
    // ----- Shell -----
    GrammarConfig {
        name: "shell",
        grammar_path: "grammars/shell.json",
        input_paths: &["input/shell.txt"],
        table_path: "table/shell_glr_table.csv",
        lr_table_path: "table/shell_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- SQL -----
    GrammarConfig {
        name: "sql",
        grammar_path: "grammars/sql.json",
        input_paths: &["input/sql.txt"],
        table_path: "table/sql_glr_table.csv",
        lr_table_path: "table/sql_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- TinyC -----
    GrammarConfig {
        name: "tinyc",
        grammar_path: "grammars/tinyc.json",
        input_paths: &["input/tinyc_small.txt", "input/tinyc_large.txt"],
        table_path: "table/tinyc_glr_table.csv",
        lr_table_path: "table/tinyc_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
    // ----- TinyC LR(1) -----
    GrammarConfig {
        name: "tinyc_lr",
        grammar_path: "grammars/lr_tinyc.json",
        input_paths: &["input/tinyc_lr_large.txt"],
        table_path: "table/lr_tinyc_glr_table.csv",
        lr_table_path: "table/lr_tinyc_lr_table.csv",
        generate_table: false,
        parsers: ALL_PARSERS,
    },
];

const WARMUP_ITERATIONS: u32 = 1;
const MIN_ITERATIONS: u32 = 10;
const MAX_ITERATIONS: u32 = 20;
const TARGET_TIME: Duration = Duration::from_millis(500);
const TIMEOUT_THRESHOLD: f64 = 1_000_000_000.0; // 1 seconds in ns

#[derive(Clone)]
struct BenchmarkResult {
    parser: String,
    input_length: usize,
    token_count: usize,
    median_time_ns: f64,
    mad_ns: f64,
    peak_memory_bytes: usize,
    iterations: u32,
    recognized: bool,
    parse_correct: bool,
    status: String, // "OK", "CONFLICT", "TIMEOUT"
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{},{},{},{}",
            self.parser,
            self.input_length,
            self.token_count,
            self.median_time_ns,
            self.mad_ns,
            self.peak_memory_bytes,
            self.iterations,
            self.recognized,
            self.parse_correct,
            self.status
        )
    }

    fn conflict(parser_name: &str, input_length: usize, token_count: usize) -> Self {
        BenchmarkResult {
            parser: parser_name.to_string(),
            input_length,
            token_count,
            median_time_ns: 0.0,
            mad_ns: 0.0,
            peak_memory_bytes: 0,
            iterations: 0,
            recognized: false,
            parse_correct: false,
            status: "CONFLICT".to_string(),
        }
    }

    fn timeout(parser_name: &str, input_length: usize, token_count: usize) -> Self {
        BenchmarkResult {
            parser: parser_name.to_string(),
            input_length,
            token_count,
            median_time_ns: 0.0,
            mad_ns: 0.0,
            peak_memory_bytes: 0,
            iterations: 0,
            recognized: false,
            parse_correct: false,
            status: "TIMEOUT".to_string(),
        }
    }
}

// ============================================================================
// Measurement Functions
// ============================================================================

/// Measure peak memory usage during parsing using a sampling thread
fn measure_peak_memory<F>(mut parse_fn: F) -> usize
where
    F: FnMut() -> Option<ParseTree>,
{
    let start_mem = memory_stats().map(|u| u.physical_mem).unwrap_or(0);

    let peak_mem = Arc::new(AtomicUsize::new(start_mem));
    let stop_signal = Arc::new(AtomicBool::new(false));

    let t_peak = peak_mem.clone();
    let t_stop = stop_signal.clone();

    // Spawn sampler thread (1ms interval)
    let sampler = thread::spawn(move || {
        while !t_stop.load(Ordering::Relaxed) {
            if let Some(usage) = memory_stats() {
                t_peak.fetch_max(usage.physical_mem, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Run the parser (single iteration)
    let _ = black_box(parse_fn());

    // Stop sampler
    stop_signal.store(true, Ordering::Relaxed);
    let _ = sampler.join();

    let peak = peak_mem.load(Ordering::Relaxed);

    // Return approximate delta (peak - start)
    // Note: This assumes single-threaded parser execution dominates memory usage
    if peak > start_mem {
        peak - start_mem
    } else {
        0
    }
}

/// Measure a parsing function with statistical rigor
fn measure<F>(mut parse_fn: F) -> (f64, f64, u32)
where
    F: FnMut() -> Option<ParseTree>,
{
    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = black_box(parse_fn());
        // if let Some(tree) = result {
        //     println!("{}", tree.display());
        // }
    }

    let mut times: Vec<f64> = Vec::new();
    let mut iterations = 0u32;

    let start_measure = Instant::now();
    loop {
        if iterations >= MAX_ITERATIONS {
            break;
        }
        if iterations >= MIN_ITERATIONS && start_measure.elapsed() >= TARGET_TIME {
            break;
        }

        let start = Instant::now();
        let _ = black_box(parse_fn());
        let elapsed = start.elapsed().as_nanos() as f64;
        times.push(elapsed);
        iterations += 1;
    }

    if times.is_empty() {
        return (0.0, 0.0, 0);
    }

    // Sort for median calculation
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate median
    let median = if times.len() % 2 == 0 {
        (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
    } else {
        times[times.len() / 2]
    };

    // Calculate MAD (Median Absolute Deviation)
    let mut deviations: Vec<f64> = times.iter().map(|t| (t - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if deviations.len() % 2 == 0 {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    (median, mad, iterations)
}

/// Check recognition and parse correctness in a single parse call.
/// Returns (recognized, parse_correct).
fn check_correctness<F>(mut parse_fn: F, expected: &str) -> (bool, bool)
where
    F: FnMut() -> Option<ParseTree>,
{
    match parse_fn() {
        Some(tree) => (true, tree.to_flat_string() == expected),
        None => (false, false),
    }
}

// ============================================================================
// Main Benchmark Logic
// ============================================================================

fn run_benchmarks(config: &GrammarConfig) -> std::io::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Benchmarking: {} grammar", config.name);
    println!("  Grammar: {}", config.grammar_path);
    println!("  Inputs:  {:?}", config.input_paths);
    println!("{}", "=".repeat(60));

    // Setup CSV file and write header
    fs::create_dir_all("results")?;
    let filename = format!("results/benchmark_{}.csv", config.name);
    let mut csv_file = File::create(&filename)?;
    writeln!(
        csv_file,
        "parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,recognized,parse_correct,status"
    )?;

    // Load grammar
    let grammar =
        grammars::load_grammar_from_file(config.grammar_path).expect("Failed to load grammar");
    let cnf_grammar = grammar.to_cnf();

    println!("✓ Grammar loaded.");
    // Setup GLR and LR tables
    if config.generate_table {
        println!("✓ Generating GLR and LR tables...");
        let table_generator = table_generator::TableGenerator::new(&grammar);
        table_generator
            .export_to_csv_numeric(config.table_path)
            .expect("Failed to export GLR table");
        table_generator
            .export_lr1_to_csv(config.lr_table_path)
            .expect("Failed to export LR table");
        println!("✓ GLR and LR tables generated.");
    } else {
        println!("✓ Skipping table generation (using existing tables)...");
    }

    println!("\n✓ Writing results to: {}", filename);
    let mut rnglr = glr::RnglrParser::import_table_from_csv(config.table_path)
        .expect("Failed to load RNGLR table");
    let mut brnglr = glr::BrnglrParser::import_table_from_csv(config.table_path)
        .expect("Failed to load BRNGLR table");
    let mut gll_parser = gll::GLLParser::new(&grammar);
    let mut earley = earley::EarleyParser::new(grammar.clone());
    let mut leo = earley_leo::LeoParser::new(grammar.clone());
    let mut conflict_parsers: HashSet<String> = HashSet::new();

    let ll_parser = if config.parsers.contains(&"LL") {
        let grammar_clone = grammar.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ll::LLParser::new(&grammar_clone)
        })) {
            Ok(parser) => Some(parser),
            Err(_) => {
                eprintln!("    [CONFLICT] LL parser has grammar conflicts, marking as CONFLICT");
                conflict_parsers.insert("LL".to_string());
                None
            }
        }
    } else {
        None
    };
    let lr_parser = if config.parsers.contains(&"LR") {
        let grammar_clone = grammar.clone();
        let lr_table_path = config.lr_table_path.to_string();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lr::LRParser::from_csv(&lr_table_path, &grammar_clone)
        })) {
            Ok(Ok(parser)) => Some(parser),
            Ok(Err(e)) => {
                eprintln!("    [ERROR] LR parser failed to load table: {}", e);
                conflict_parsers.insert("LR".to_string());
                None
            }
            Err(_) => {
                eprintln!("    [CONFLICT] LR parser has grammar conflicts, marking as CONFLICT");
                conflict_parsers.insert("LR".to_string());
                None
            }
        }
    } else {
        None
    };

    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    // Load inputs from all input files
    let mut all_input_lines: Vec<String> = Vec::new();
    for input_path in config.input_paths {
        let input_content = fs::read_to_string(input_path)
            .unwrap_or_else(|_| panic!("Failed to read input file: {}", input_path));
        for line in input_content.lines() {
            if !line.trim().is_empty() {
                all_input_lines.push(line.to_string());
            }
        }
    }

    // Sort by length for nicer plotting
    all_input_lines.sort_by_key(|l| l.len());

    let lines: Vec<&str> = all_input_lines.iter().map(|s| s.as_str()).collect();

    println!(
        "Found {} inputs (lengths: {} to {} bytes)",
        lines.len(),
        lines.first().map(|l| l.len()).unwrap_or(0),
        lines.last().map(|l| l.len()).unwrap_or(0)
    );

    let mut failed_parsers: HashSet<String> = HashSet::new();

    for (idx, line) in lines.iter().enumerate() {
        let input_len = line.len();

        // Tokenize
        let tokens = match grammar.tokenize(line) {
            Some(t) => t,
            None => {
                eprintln!("  [SKIP] Input #{}: Failed to tokenize", idx + 1);
                continue;
            }
        };
        let token_count = tokens.len();
        let cnf_tokens = cnf_grammar.tokenize(line).unwrap_or_default();
        let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        println!(
            "\n  Input #{}: {} bytes, {} tokens",
            idx + 1,
            input_len,
            token_count
        );
        println!("{}", &line[..std::cmp::min(50, line.len())].trim());
        for parser_name in config.parsers {
            // Handle conflict parsers: write CONFLICT row and skip
            if conflict_parsers.contains(*parser_name) {
                let result = BenchmarkResult::conflict(parser_name, input_len, token_count);
                println!("    [CONFLICT] {:8}: grammar has conflicts, skipping", parser_name);
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            // Handle timed-out parsers: write TIMEOUT row and skip
            if failed_parsers.contains(*parser_name) {
                let result = BenchmarkResult::timeout(parser_name, input_len, token_count);
                println!("    [TIMEOUT] {:8}: previously timed out, skipping", parser_name);
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            let result = match *parser_name {
                "Earley" => {
                    let (recognized, parse_correct) = check_correctness(|| earley.parse(tokens.clone()), line);
                    let peak_mem = measure_peak_memory(|| earley.parse(tokens.clone()));
                    let (median, mad, iters) = measure(|| earley.parse(tokens.clone()));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "Leo" => {
                    let (recognized, parse_correct) = check_correctness(|| leo.parse(tokens.clone()), line);
                    let peak_mem = measure_peak_memory(|| leo.parse(tokens.clone()));
                    let (median, mad, iters) = measure(|| leo.parse(tokens.clone()));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "GLL" => {
                    let (recognized, parse_correct) = check_correctness(|| gll_parser.parse(&tokens), line);
                    let peak_mem = measure_peak_memory(|| gll_parser.parse(&tokens));
                    let (median, mad, iters) = measure(|| gll_parser.parse(&tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "RNGLR" => {
                    let (recognized, parse_correct) = check_correctness(|| rnglr.parse(&glr_tokens), line);
                    let peak_mem = measure_peak_memory(|| rnglr.parse(&glr_tokens));
                    let (median, mad, iters) = measure(|| rnglr.parse(&glr_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "BRNGLR" => {
                    let (recognized, parse_correct) = check_correctness(|| brnglr.parse(&glr_tokens), line);
                    let peak_mem = measure_peak_memory(|| brnglr.parse(&glr_tokens));
                    let (median, mad, iters) = measure(|| brnglr.parse(&glr_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "CYK" => {
                    let (recognized, parse_correct) = check_correctness(|| cyk::parse(&cnf_grammar, &cnf_tokens), line);
                    let peak_mem = measure_peak_memory(|| cyk::parse(&cnf_grammar, &cnf_tokens));
                    let (median, mad, iters) = measure(|| cyk::parse(&cnf_grammar, &cnf_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "Valiant" => {
                    let (recognized, parse_correct) = check_correctness(|| valiant::parse(&cnf_grammar, &cnf_tokens), line);
                    let peak_mem =
                        measure_peak_memory(|| valiant::parse(&cnf_grammar, &cnf_tokens));
                    let (median, mad, iters) =
                        measure(|| valiant::parse(&cnf_grammar, &cnf_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        peak_memory_bytes: peak_mem,
                        iterations: iters,
                        recognized,
                        parse_correct,
                        status: "OK".to_string(),
                    }
                }
                "LR" => {
                    if let Some(parser) = &lr_parser {
                        let (recognized, parse_correct) = check_correctness(|| parser.parse(&glr_tokens), line);
                        let peak_mem = measure_peak_memory(|| parser.parse(&glr_tokens));
                        let (median, mad, iters) = measure(|| parser.parse(&glr_tokens));
                        BenchmarkResult {
                            parser: parser_name.to_string(),
                            input_length: input_len,
                            token_count,
                            median_time_ns: median,
                            mad_ns: mad,
                            peak_memory_bytes: peak_mem,
                            iterations: iters,
                            recognized,
                            parse_correct,
                            status: "OK".to_string(),
                        }
                    } else {
                        continue; // conflict already handled above
                    }
                }
                "LL" => {
                    if let Some(parser) = &ll_parser {
                        let (recognized, parse_correct) = check_correctness(|| parser.parse(&tokens), line);
                        let peak_mem = measure_peak_memory(|| parser.parse(&tokens));
                        let (median, mad, iters) = measure(|| parser.parse(&tokens));
                        BenchmarkResult {
                            parser: parser_name.to_string(),
                            input_length: input_len,
                            token_count,
                            median_time_ns: median,
                            mad_ns: mad,
                            peak_memory_bytes: peak_mem,
                            iterations: iters,
                            recognized,
                            parse_correct,
                            status: "OK".to_string(),
                        }
                    } else {
                        continue; // conflict already handled above
                    }
                }
                _ => continue,
            };

            let recog_status = if result.recognized { "R" } else { "✗" };
            let parse_status = if result.parse_correct { "P" } else { "✗" };
            println!(
                "    [{}|{}] {:8}: {:>12.0} ns ± {:>8.0} ns ({} iters)",
                recog_status, parse_status, result.parser, result.median_time_ns, result.mad_ns, result.iterations
            );

            if !result.recognized {
                eprintln!(
                    "    [WARN] Parser {} failed to recognize input with length {} ({} tokens)",
                    result.parser, result.input_length, result.token_count
                );
            }
            if !result.parse_correct {
                eprintln!(
                    "    [WARN] Parser {} produced incorrect parse tree for input with length {} ({} tokens)",
                    result.parser, result.input_length, result.token_count
                );
            }

            // Write result to CSV immediately
            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?; // Ensure data is written to disk immediately

            // If parser took too long, mark it as failed for future iterations
            if result.median_time_ns > TIMEOUT_THRESHOLD {
                println!(
                    "    [TIMEOUT] {} exceeded 5s threshold, skipping larger inputs",
                    result.parser
                );
                failed_parsers.insert(result.parser);
            }
        }
    }

    Ok(())
}

fn run_main() {
    println!("Parser Benchmark Tool");
    println!("=====================");
    println!("Generating CSV data for plotting parsing time vs input length\n");
    println!("Configuration:");
    println!("  Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("  Min iterations: {}", MIN_ITERATIONS);
    println!("  Max iterations: {}", MAX_ITERATIONS);
    println!("  Target time: {:?}", TARGET_TIME);
    // println!("  Parsers: {:?}", PARSERS); // Parsers are now per-config

    for config in CONFIGS {
        if let Err(e) = run_benchmarks(config) {
            eprintln!("Error running benchmarks: {}", e);
        }
    }

    println!("\n✓ Benchmarking complete!");
}

fn main() {
    // Use a larger stack size to handle deep recursion in parsers
    // Default stack is ~2MB, we use 128MB to handle complex grammars like CSS
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024) // 128 MB stack
        .spawn(run_main)
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}
