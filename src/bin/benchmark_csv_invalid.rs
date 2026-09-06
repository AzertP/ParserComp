//! Benchmarking tool for invalid scannerless inputs.
//!
//! Usage:
//!   cargo run --release --bin benchmark_csv_invalid
//!
//! Output:
//!   Creates a CSV file in results/benchmark_csv_invalid/ with columns:
//!   parser, input_length, tokens, median_time_ns, mad_ns, iterations, recognized, reject_correct
use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::earley;
use parser_comparison::parsers::gll::ll;
use parser_comparison::parsers::glr::lr;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{cyk, earley_leo, gll, glr, valiant};
use std::collections::HashSet;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
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

const ALL_PARSERS: &[&str] = &[
    "Leo", "GLL", "RNGLR", "BRNGLR", "CYK", "LL", "LR", "Valiant",
];

/// Parsers excluding CYK and Valiant, for grammars where those parsers
/// are impractical (exceed 1 s below 200 tokens in the manuscript).
const FAST_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR", "LL", "LR"];

const CONFIGS: &[GrammarConfig] = &[
    // ----- ANSI C -----
    GrammarConfig {
        name: "ansi_c",
        grammar_path: "grammars/ansi_c.json",
        input_paths: &["input/ansi_c_invalid.txt"],
        table_path: "table/ansi_c_glr_table.csv",
        lr_table_path: "table/ansi_c_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- Bool -----
    GrammarConfig {
        name: "bool",
        grammar_path: "grammars/bool.json",
        input_paths: &["input/bool_invalid.txt"],
        table_path: "table/bool_glr_table.csv",
        lr_table_path: "table/bool_lr_table.csv",
        generate_table: true,
        parsers: FAST_PARSERS,
    },
    // ----- Calculator -----
    GrammarConfig {
        name: "calc",
        grammar_path: "grammars/calc.json",
        input_paths: &["input/expr_invalid.txt"],
        table_path: "table/calc_glr_table.csv",
        lr_table_path: "table/calc_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- Calculator LL(1) -----
    GrammarConfig {
        name: "calc_ll1",
        grammar_path: "grammars/ll1_calc.json",
        input_paths: &["input/expr_invalid.txt"],
        table_path: "table/calc_ll1_glr_table.csv",
        lr_table_path: "table/calc_ll1_lr_table.csv",
        generate_table: true,
        parsers: FAST_PARSERS,
    },
    // ----- Expr -----
    GrammarConfig {
        name: "expr",
        grammar_path: "grammars/expr.json",
        input_paths: &["input/expr_invalid.txt"],
        table_path: "table/expr_glr_table.csv",
        lr_table_path: "table/expr_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- Expr Ambiguous -----
    GrammarConfig {
        name: "expr_ambi",
        grammar_path: "grammars/expr_ambi.json",
        input_paths: &["input/expr_invalid.txt"],
        table_path: "table/expr_ambi_glr_table.csv",
        lr_table_path: "table/expr_ambi_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- C++ -----
    GrammarConfig {
        name: "cpp",
        grammar_path: "grammars/cpp.json",
        input_paths: &["input/cpp_invalid.txt"],
        table_path: "table/cpp_glr_table.csv",
        lr_table_path: "table/cpp_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- CSS -----
    GrammarConfig {
        name: "css",
        grammar_path: "grammars/css.json",
        input_paths: &["input/css_invalid.txt"],
        table_path: "table/css_glr_table.csv",
        lr_table_path: "table/css_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- HTML -----
    GrammarConfig {
        name: "html",
        grammar_path: "grammars/html.json",
        input_paths: &["input/html_invalid.txt"],
        table_path: "table/html_glr_table.csv",
        lr_table_path: "table/html_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- Json -----
    GrammarConfig {
        name: "json",
        grammar_path: "grammars/json.json",
        input_paths: &["input/json_invalid.txt"],
        table_path: "table/json_glr_table.csv",
        lr_table_path: "table/json_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // ----- Json LL(1) -----
    GrammarConfig {
        name: "json_ll1",
        grammar_path: "grammars/ll1_json.json",
        input_paths: &["input/json_invalid.txt"],
        table_path: "table/json_ll1_glr_table.csv",
        lr_table_path: "table/json_ll1_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- Json LR(1) -----
    GrammarConfig {
        name: "json_lr",
        grammar_path: "grammars/lr_json.json",
        input_paths: &["input/json_invalid.txt"],
        table_path: "table/lr_json_glr_table.csv",
        lr_table_path: "table/lr_json_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- Json Ambiguous -----
    GrammarConfig {
        name: "json_ambi",
        grammar_path: "grammars/json_ambi.json",
        input_paths: &["input/json_invalid.txt"],
        table_path: "table/json_ambi_glr_table.csv",
        lr_table_path: "table/json_ambi_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- Java -----
    GrammarConfig {
        name: "java",
        grammar_path: "grammars/jsl18.json",
        input_paths: &["input/java_invalid.txt"],
        table_path: "table/java_glr_table.csv",
        lr_table_path: "table/java_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- Pascal -----
    GrammarConfig {
        name: "pascal",
        grammar_path: "grammars/pascal.json",
        input_paths: &["input/pascal_invalid.txt"],
        table_path: "table/pascal_glr_table.csv",
        lr_table_path: "table/pascal_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- TinyPascal -----
    GrammarConfig {
        name: "tinypascal",
        grammar_path: "grammars/ll1_tinypascal.json",
        input_paths: &["input/tinypascal_invalid.txt"],
        table_path: "table/tinypascal_glr_table.csv",
        lr_table_path: "table/tinypascal_lr_table.csv",
        generate_table: true,
        parsers: FAST_PARSERS,
    },
    // // ----- S-exp -----
    GrammarConfig {
        name: "sexp",
        grammar_path: "grammars/sexp.json",
        input_paths: &["input/sexp_invalid.txt"],
        table_path: "table/sexp_glr_table.csv",
        lr_table_path: "table/sexp_lr_table.csv",
        generate_table: true,
        parsers: FAST_PARSERS,
    },
    // ----- S-exp LL(1) -----
    GrammarConfig {
        name: "sexp_ll1",
        grammar_path: "grammars/ll1_sexp.json",
        input_paths: &["input/sexp_invalid.txt"],
        table_path: "table/sexp_ll1_glr_table.csv",
        lr_table_path: "table/sexp_ll1_lr_table.csv",
        generate_table: true,
        parsers: FAST_PARSERS,
    },
    // // ----- Shell -----
    GrammarConfig {
        name: "shell",
        grammar_path: "grammars/shell.json",
        input_paths: &["input/shell_invalid.txt"],
        table_path: "table/shell_glr_table.csv",
        lr_table_path: "table/shell_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- SQL -----
    GrammarConfig {
        name: "sql",
        grammar_path: "grammars/sql.json",
        input_paths: &["input/sql_invalid.txt"],
        table_path: "table/sql_glr_table.csv",
        lr_table_path: "table/sql_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // ----- TinyC -----
    GrammarConfig {
        name: "tinyc",
        grammar_path: "grammars/tinyc.json",
        input_paths: &["input/tinyc_invalid.txt"],
        table_path: "table/tinyc_glr_table.csv",
        lr_table_path: "table/tinyc_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
    // // // ----- TinyC LR(1) -----
    GrammarConfig {
        name: "tinyc_lr",
        grammar_path: "grammars/lr_tinyc.json",
        input_paths: &["input/tinyc_lr_invalid.txt"],
        table_path: "table/lr_tinyc_glr_table.csv",
        lr_table_path: "table/lr_tinyc_lr_table.csv",
        generate_table: false,
        parsers: FAST_PARSERS,
    },
];

const WARMUP_ITERATIONS: u32 = 1;
const TIMED_ITERATIONS: u32 = 10;
const PARSE_TIMEOUT: Duration = Duration::from_secs(1);
const TIMEOUT_EXIT_CODE: i32 = 124;
const PAIR_WORKER_ARG: &str = "--pair-worker";
const WORKER_RESULT_PREFIX: &str = "WORKER_RESULT\t";
const RESULT_DIR: &str = "results/benchmark_csv_invalid";

fn output_path(config_name: &str) -> String {
    format!("{}/benchmark_{}_invalid.csv", RESULT_DIR, config_name)
}

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
    reject_correct: bool,
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
            self.reject_correct,
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
            reject_correct: false,
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
            reject_correct: false,
            status: "TIMEOUT".to_string(),
        }
    }

    fn from_csv_row(row: &str) -> Result<Self, String> {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() != 10 {
            return Err(format!(
                "expected 10 worker result fields, found {}",
                fields.len()
            ));
        }
        Ok(BenchmarkResult {
            parser: fields[0].to_string(),
            input_length: fields[1]
                .parse()
                .map_err(|e| format!("invalid input length: {e}"))?,
            token_count: fields[2]
                .parse()
                .map_err(|e| format!("invalid token count: {e}"))?,
            median_time_ns: fields[3]
                .parse()
                .map_err(|e| format!("invalid median time: {e}"))?,
            mad_ns: fields[4].parse().map_err(|e| format!("invalid MAD: {e}"))?,
            peak_memory_bytes: fields[5]
                .parse()
                .map_err(|e| format!("invalid peak memory: {e}"))?,
            iterations: fields[6]
                .parse()
                .map_err(|e| format!("invalid iteration count: {e}"))?,
            recognized: fields[7]
                .parse()
                .map_err(|e| format!("invalid recognized flag: {e}"))?,
            reject_correct: fields[8]
                .parse()
                .map_err(|e| format!("invalid rejection flag: {e}"))?,
            status: fields[9].to_string(),
        })
    }
}

fn worker_result_from_output(
    exit_code: Option<i32>,
    stdout: &[u8],
    parser_name: &str,
    input_length: usize,
    token_count: usize,
) -> std::io::Result<BenchmarkResult> {
    if exit_code == Some(TIMEOUT_EXIT_CODE) {
        return Ok(BenchmarkResult::timeout(
            parser_name,
            input_length,
            token_count,
        ));
    }
    if exit_code != Some(0) {
        return Err(std::io::Error::other(format!(
            "benchmark worker exited with code {exit_code:?}"
        )));
    }
    let output = std::str::from_utf8(stdout)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let payload = output
        .lines()
        .find_map(|line| line.strip_prefix(WORKER_RESULT_PREFIX))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "benchmark worker produced no result row",
            )
        })?;
    BenchmarkResult::from_csv_row(payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ============================================================================
// Measurement Functions
// ============================================================================

/// Measure peak memory usage during parsing using a sampling thread
fn measure_peak_memory_with_result<F>(mut parse_fn: F) -> (Option<ParseTree>, usize)
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
    let parse_result = black_box(parse_fn());

    // Stop sampler
    stop_signal.store(true, Ordering::Relaxed);
    let _ = sampler.join();

    let peak = peak_mem.load(Ordering::Relaxed);

    // Return approximate delta (peak - start)
    // Note: This assumes single-threaded parser execution dominates memory usage
    (parse_result, peak.saturating_sub(start_mem))
}

fn run_with_hard_timeout<T, F>(timeout: Duration, operation: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let watchdog = thread::spawn(move || {
        if matches!(
            done_rx.recv_timeout(timeout),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ) {
            std::process::exit(TIMEOUT_EXIT_CODE);
        }
    });
    let start = Instant::now();
    let result = operation();
    let elapsed = start.elapsed();
    let _ = done_tx.send(());
    watchdog.join().expect("timeout watchdog panicked");
    (result, elapsed)
}

/// Measure a parsing function with statistical rigor
fn measure<F>(mut parse_fn: F) -> (f64, f64, u32)
where
    F: FnMut() -> Option<ParseTree>,
{
    let mut times: Vec<f64> = Vec::with_capacity(TIMED_ITERATIONS as usize);
    for _ in 0..TIMED_ITERATIONS {
        let (_, elapsed) = run_with_hard_timeout(PARSE_TIMEOUT, || black_box(parse_fn()));
        times.push(elapsed.as_nanos() as f64);
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

    (median, mad, times.len() as u32)
}

/// Check whether an invalid input is rejected in a single parse call.
/// Returns (recognized, reject_correct).
fn benchmark_parser<F>(
    parser_name: &str,
    input_length: usize,
    token_count: usize,
    mut parse_fn: F,
) -> BenchmarkResult
where
    F: FnMut() -> Option<ParseTree>,
{
    let ((warmup_result, peak_memory_bytes), _) = run_with_hard_timeout(PARSE_TIMEOUT, || {
        measure_peak_memory_with_result(&mut parse_fn)
    });
    let (recognized, reject_correct) = match warmup_result {
        Some(_) => (true, false),
        None => (false, true),
    };
    let (median_time_ns, mad_ns, iterations) = measure(parse_fn);
    BenchmarkResult {
        parser: parser_name.to_string(),
        input_length,
        token_count,
        median_time_ns,
        mad_ns,
        peak_memory_bytes,
        iterations,
        recognized,
        reject_correct,
        status: "OK".to_string(),
    }
}

fn run_pair_worker(
    config: &GrammarConfig,
    parser_name: &str,
    input: &str,
) -> Result<BenchmarkResult, String> {
    let grammar = grammars::load_grammar_from_file(config.grammar_path)
        .map_err(|e| format!("failed to load grammar: {e}"))?;
    let tokens = grammar
        .tokenize(input)
        .ok_or_else(|| "failed to tokenize worker input".to_string())?;
    let input_length = input.len();
    let token_count = tokens.len();

    match parser_name {
        "Earley" => {
            let mut parser = earley::EarleyParser::new(grammar.clone());
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || parser.parse(tokens.clone()),
            ))
        }
        "Leo" => {
            let mut parser = earley_leo::LeoParser::new(grammar.clone());
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || parser.parse(tokens.clone()),
            ))
        }
        "GLL" => {
            let mut parser = gll::GLLParser::new(&grammar);
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || parser.parse_one(&tokens),
            ))
        }
        "RNGLR" => {
            let mut parser = glr::RnglrParser::import_table_from_csv(config.table_path)
                .map_err(|e| format!("failed to load RNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            let ids: Vec<i32> = tokens.iter().map(|&token| (token + 1) as i32).collect();
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || parser.parse(&ids),
            ))
        }
        "BRNGLR" => {
            let mut parser = glr::BrnglrParser::import_table_from_csv(config.table_path)
                .map_err(|e| format!("failed to load BRNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            let ids: Vec<i32> = tokens.iter().map(|&token| (token + 1) as i32).collect();
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || parser.parse(&ids),
            ))
        }
        "CYK" => {
            let cnf = grammar.to_cnf();
            let ids = cnf.tokenize(input).unwrap_or_default();
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || cyk::parse(&cnf, &ids),
            ))
        }
        "Valiant" => {
            let cnf = grammar.to_cnf();
            let ids = cnf.tokenize(input).unwrap_or_default();
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || valiant::parse(&cnf, &ids),
            ))
        }
        "LL" => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ll::LLParser::new(&grammar)
        })) {
            Ok(parser) => Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                || parser.parse(&tokens),
            )),
            Err(_) => Ok(BenchmarkResult::conflict(
                parser_name,
                input_length,
                token_count,
            )),
        },
        "LR" => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lr::LRParser::from_csv(config.lr_table_path, &grammar)
        })) {
            Ok(Ok(parser)) => {
                let ids: Vec<i32> = tokens.iter().map(|&token| (token + 1) as i32).collect();
                Ok(benchmark_parser(
                    parser_name,
                    input_length,
                    token_count,
                    || parser.parse(&ids),
                ))
            }
            Ok(Err(_)) | Err(_) => Ok(BenchmarkResult::conflict(
                parser_name,
                input_length,
                token_count,
            )),
        },
        _ => Err(format!("unknown parser: {parser_name}")),
    }
}

fn pair_worker_main(args: &[String]) -> i32 {
    if args.len() != 4 {
        eprintln!("Usage: {} {PAIR_WORKER_ARG} <config> <parser>", args[0]);
        return 2;
    }
    let Some(config) = CONFIGS.iter().find(|config| config.name == args[2]) else {
        eprintln!("Unknown benchmark configuration: {}", args[2]);
        return 2;
    };
    let mut input = String::new();
    if let Err(error) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("Failed to read worker input: {error}");
        return 2;
    }
    match run_pair_worker(config, &args[3], &input) {
        Ok(result) => {
            println!("{WORKER_RESULT_PREFIX}{}", result.to_csv_row());
            0
        }
        Err(error) => {
            eprintln!("Benchmark worker failed: {error}");
            1
        }
    }
}

fn run_pair_in_worker(
    config: &GrammarConfig,
    parser_name: &str,
    input: &str,
    input_length: usize,
    token_count: usize,
) -> std::io::Result<BenchmarkResult> {
    let mut child = Command::new(std::env::current_exe()?)
        .args([PAIR_WORKER_ARG, config.name, parser_name])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("worker stdin was not piped"))?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    worker_result_from_output(
        output.status.code(),
        &output.stdout,
        parser_name,
        input_length,
        token_count,
    )
    .map_err(|error| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        std::io::Error::new(error.kind(), format!("{error}; worker stderr: {stderr}"))
    })
}

// ============================================================================
// Main Benchmark Logic
// ============================================================================

fn run_benchmarks(config: &GrammarConfig) -> std::io::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Benchmarking invalid inputs: {} grammar", config.name);
    println!("  Grammar: {}", config.grammar_path);
    println!("  Inputs:  {:?}", config.input_paths);
    println!("{}", "=".repeat(60));

    // Setup CSV file and write header
    fs::create_dir_all(RESULT_DIR)?;
    let filename = output_path(config.name);
    let mut csv_file = File::create(&filename)?;
    writeln!(
        csv_file,
        "parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,recognized,reject_correct,status"
    )?;

    // Load grammar
    let grammar =
        grammars::load_grammar_from_file(config.grammar_path).expect("Failed to load grammar");
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
    let mut conflict_parsers: HashSet<String> = HashSet::new();

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
        "Found {} invalid inputs (lengths: {} to {} bytes)",
        lines.len(),
        lines.first().map(|l| l.len()).unwrap_or(0),
        lines.last().map(|l| l.len()).unwrap_or(0)
    );

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
                println!(
                    "    [CONFLICT] {:8}: grammar has conflicts, skipping",
                    parser_name
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            let result = run_pair_in_worker(config, parser_name, line, input_len, token_count)?;

            if result.status == "CONFLICT" {
                conflict_parsers.insert(result.parser.clone());
                println!(
                    "    [CONFLICT] {:8}: grammar has conflicts, skipping",
                    result.parser
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            if result.status == "TIMEOUT" {
                println!(
                    "    [TIMEOUT] {:8}: invocation exceeded {:?}",
                    result.parser, PARSE_TIMEOUT
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            let outcome_status = if result.recognized {
                "ACCEPT"
            } else {
                "REJECT"
            };
            let reject_status = if result.reject_correct { "OK" } else { "FAIL" };
            println!(
                "    [{:6}|{}] {:8}: {:>12.0} ns ± {:>8.0} ns ({} iters)",
                outcome_status,
                reject_status,
                result.parser,
                result.median_time_ns,
                result.mad_ns,
                result.iterations
            );

            if result.recognized {
                eprintln!(
                    "    [WARN] Parser {} accepted invalid input with length {} ({} tokens)",
                    result.parser, result.input_length, result.token_count
                );
            }
            if !result.reject_correct {
                eprintln!(
                    "    [WARN] Parser {} did not reject invalid input with length {} ({} tokens)",
                    result.parser, result.input_length, result.token_count
                );
            }

            // Write result to CSV immediately
            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?; // Ensure data is written to disk immediately
        }
    }

    Ok(())
}

fn run_main() {
    println!("Invalid Input Parser Benchmark Tool");
    println!("=====================");
    println!("Generating CSV data for rejection behavior vs input length\n");
    println!("Configuration:");
    println!("  Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("  Timed iterations: {}", TIMED_ITERATIONS);
    println!("  Per-invocation timeout: {:?}", PARSE_TIMEOUT);
    // println!("  Parsers: {:?}", PARSERS); // Parsers are now per-config

    for config in CONFIGS {
        if let Err(e) = run_benchmarks(config) {
            eprintln!("Error running benchmarks: {}", e);
        }
    }

    println!("\n✓ Benchmarking complete!");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some(PAIR_WORKER_ARG) {
        let exit_code = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(move || pair_worker_main(&args))
            .expect("Failed to spawn worker thread with larger stack")
            .join()
            .expect("Worker thread panicked");
        std::process::exit(exit_code);
    }

    // Use a larger stack size to handle deep recursion in parsers
    // Default stack is ~2MB, we use 128MB to handle complex grammars like CSS
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024) // 128 MB stack
        .spawn(run_main)
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT_CHILD_ENV: &str = "BENCHMARK_CSV_INVALID_TIMEOUT_CHILD";

    #[test]
    fn output_path_uses_benchmark_csv_invalid_result_folder() {
        assert_eq!(
            output_path("json"),
            "results/benchmark_csv_invalid/benchmark_json_invalid.csv"
        );
    }

    #[test]
    fn measure_runs_exactly_ten_timed_iterations() {
        let mut calls = 0;
        let (_, _, iterations) = measure(|| {
            calls += 1;
            None
        });

        assert_eq!(iterations, 10);
        assert_eq!(calls, 10);
    }

    #[test]
    fn benchmark_pair_runs_one_warmup_and_ten_timed_iterations() {
        let mut calls = 0;
        let result = benchmark_parser("Test", 4, 2, || {
            calls += 1;
            None
        });
        assert_eq!(calls, 11);
        assert_eq!(result.iterations, 10);
        assert!(result.reject_correct);
    }

    #[test]
    fn hard_timeout_terminates_worker_process() {
        if std::env::var_os(TIMEOUT_CHILD_ENV).is_some() {
            let _ = run_with_hard_timeout(Duration::from_millis(25), || {
                thread::sleep(Duration::from_secs(5));
            });
            panic!("slow operation completed instead of timing out");
        }
        let started = Instant::now();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::hard_timeout_terminates_worker_process"])
            .env(TIMEOUT_CHILD_ENV, "1")
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(TIMEOUT_EXIT_CODE));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn worker_exit_124_becomes_timeout_result() {
        let result = worker_result_from_output(Some(124), b"", "Leo", 42, 17).unwrap();
        assert_eq!(result.parser, "Leo");
        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, "TIMEOUT");
    }
}
