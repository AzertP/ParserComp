//! Benchmarking tool for source-file inputs.
//!
//! Supports two input modes:
//!   - `TextLines`: old-style .txt files where each non-empty line is one input string
//!   - `CodeFiles`: a directory of source files where each file is read whole as one input
//!
//! CYK and Valiant are excluded. Only Leo, GLL, RNGLR, BRNGLR, LL, LR are benchmarked.
//!
//! Usage:
//!   cargo run --release --bin benchmark_code
//!
//! Output:
//!   Creates CSV files in results/benchmark_code/ with columns:
//!   parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,
//!   iterations,recognized,parse_correct,status,source_file
use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{earley_leo, gll, glr};
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

/// How inputs are sourced for a grammar config.
enum InputSource {
    /// Old method: one or more .txt files; each non-empty line is a separate input string.
    #[allow(dead_code)]
    TextLines(&'static [&'static str]),
    /// New method: a directory of source files; each file's full content is one input string.
    CodeFiles(&'static str),
}

struct GrammarConfig {
    name: &'static str,
    grammar_path: &'static str,
    input_source: InputSource,
    table_path: &'static str,
    lr_table_path: &'static str,
    generate_table: bool,
    parsers: &'static [&'static str],
}

/// Parsers for ws-aware whole-file benchmarks (CYK, Valiant, LL, LR excluded).
const WS_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];

fn configs() -> Vec<GrammarConfig> {
    vec![
        GrammarConfig {
            name: "c_ws",
            grammar_path: "grammars/c_ws.json",
            input_source: InputSource::CodeFiles("input/code/C"),
            table_path: "table/c_ws_glr_table.csv",
            lr_table_path: "table/c_ws_lr_table.csv",
            generate_table: false,
            parsers: WS_PARSERS,
        },
        GrammarConfig {
            name: "cpp_ws",
            grammar_path: "grammars/cpp_ws.json",
            input_source: InputSource::CodeFiles("input/code/C++"),
            table_path: "table/cpp_ws_glr_table.csv",
            lr_table_path: "table/cpp_ws_lr_table.csv",
            generate_table: false,
            parsers: WS_PARSERS,
        },
        GrammarConfig {
            name: "java_ws",
            grammar_path: "grammars/jsl18_ws.json",
            input_source: InputSource::CodeFiles("input/code/Java"),
            table_path: "table/java_ws_glr_table.csv",
            lr_table_path: "table/java_ws_lr_table.csv",
            generate_table: false,
            parsers: WS_PARSERS,
        },
    ]
}

const WARMUP_ITERATIONS: u32 = 1;
const TIMED_ITERATIONS: u32 = 10;
const PARSE_TIMEOUT: Duration = Duration::from_secs(1);
const TIMEOUT_EXIT_CODE: i32 = 124;
const PAIR_WORKER_ARG: &str = "--pair-worker";
const WORKER_RESULT_PREFIX: &str = "WORKER_RESULT\t";
const RESULT_DIR: &str = "results/benchmark_code";

fn output_path(config_name: &str) -> String {
    format!("{}/benchmark_{}.csv", RESULT_DIR, config_name)
}

// ============================================================================
// Benchmark Result
// ============================================================================

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
    status: String,
    source_file: String,
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{},{},{},{},{}",
            self.parser,
            self.input_length,
            self.token_count,
            self.median_time_ns,
            self.mad_ns,
            self.peak_memory_bytes,
            self.iterations,
            self.recognized,
            self.parse_correct,
            self.status,
            self.source_file,
        )
    }

    fn timeout(
        parser_name: &str,
        input_length: usize,
        token_count: usize,
        source_file: &str,
    ) -> Self {
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
            source_file: source_file.to_string(),
        }
    }

    fn parse_fail(
        parser_name: &str,
        input_length: usize,
        token_count: usize,
        source_file: &str,
    ) -> Self {
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
            status: "PARSE_FAIL".to_string(),
            source_file: source_file.to_string(),
        }
    }

    fn from_csv_row(row: &str) -> Result<Self, String> {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() != 11 {
            return Err(format!(
                "expected 11 worker result fields, found {}",
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
            parse_correct: fields[8]
                .parse()
                .map_err(|e| format!("invalid parse-correct flag: {e}"))?,
            status: fields[9].to_string(),
            source_file: fields[10].to_string(),
        })
    }
}

fn worker_result_from_output(
    exit_code: Option<i32>,
    stdout: &[u8],
    parser_name: &str,
    input_length: usize,
    token_count: usize,
    source_file: &str,
) -> std::io::Result<BenchmarkResult> {
    if exit_code == Some(TIMEOUT_EXIT_CODE) {
        return Ok(BenchmarkResult::timeout(
            parser_name,
            input_length,
            token_count,
            source_file,
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

fn measure_peak_memory_with_result<F>(mut parse_fn: F) -> (Option<ParseTree>, usize)
where
    F: FnMut() -> Option<ParseTree>,
{
    let start_mem = memory_stats().map(|u| u.physical_mem).unwrap_or(0);

    let peak_mem = Arc::new(AtomicUsize::new(start_mem));
    let stop_signal = Arc::new(AtomicBool::new(false));

    let t_peak = peak_mem.clone();
    let t_stop = stop_signal.clone();

    let sampler = thread::spawn(move || {
        while !t_stop.load(Ordering::Relaxed) {
            if let Some(usage) = memory_stats() {
                t_peak.fetch_max(usage.physical_mem, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    let parse_result = black_box(parse_fn());

    stop_signal.store(true, Ordering::Relaxed);
    let _ = sampler.join();

    let peak = peak_mem.load(Ordering::Relaxed);
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

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if times.len() % 2 == 0 {
        (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
    } else {
        times[times.len() / 2]
    };

    let mut deviations: Vec<f64> = times.iter().map(|t| (t - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if deviations.len() % 2 == 0 {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    (median, mad, times.len() as u32)
}

fn benchmark_parser<F>(
    parser_name: &str,
    input_length: usize,
    token_count: usize,
    source_file: &str,
    expected: &str,
    mut parse_fn: F,
) -> BenchmarkResult
where
    F: FnMut() -> Option<ParseTree>,
{
    let ((warmup_result, peak_memory_bytes), _) = run_with_hard_timeout(PARSE_TIMEOUT, || {
        measure_peak_memory_with_result(&mut parse_fn)
    });
    let (recognized, parse_correct) = match warmup_result {
        Some(tree) => (true, tree.to_flat_string() == expected),
        None => (false, false),
    };
    let (median_time_ns, mad_ns, iterations) = measure(parse_fn);
    if !recognized {
        let mut result =
            BenchmarkResult::parse_fail(parser_name, input_length, token_count, source_file);
        result.median_time_ns = median_time_ns;
        result.mad_ns = mad_ns;
        result.peak_memory_bytes = peak_memory_bytes;
        result.iterations = iterations;
        return result;
    }

    BenchmarkResult {
        parser: parser_name.to_string(),
        input_length,
        token_count,
        median_time_ns,
        mad_ns,
        peak_memory_bytes,
        iterations,
        recognized,
        parse_correct,
        status: "OK".to_string(),
        source_file: source_file.to_string(),
    }
}

fn run_pair_worker(
    config: &GrammarConfig,
    parser_name: &str,
    source_file: &str,
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
        "Leo" => {
            let mut parser = earley_leo::LeoParser::new(grammar.clone());
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                source_file,
                input,
                || parser.parse(tokens.clone()),
            ))
        }
        "GLL" => {
            let mut parser = gll::GLLParser::new(&grammar);
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                source_file,
                input,
                || parser.parse_one(&tokens),
            ))
        }
        "RNGLR" => {
            let mut parser = glr::RnglrParser::import_table_from_csv(config.table_path)
                .map_err(|e| format!("failed to load RNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            let glr_tokens: Vec<i32> = tokens.iter().map(|&token| (token + 1) as i32).collect();
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                source_file,
                input,
                || parser.parse(&glr_tokens),
            ))
        }
        "BRNGLR" => {
            let mut parser = glr::BrnglrParser::import_table_from_csv(config.table_path)
                .map_err(|e| format!("failed to load BRNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            let glr_tokens: Vec<i32> = tokens.iter().map(|&token| (token + 1) as i32).collect();
            Ok(benchmark_parser(
                parser_name,
                input_length,
                token_count,
                source_file,
                input,
                || parser.parse(&glr_tokens),
            ))
        }
        _ => Err(format!("unknown parser: {parser_name}")),
    }
}

fn pair_worker_main(args: &[String]) -> i32 {
    if args.len() != 5 {
        eprintln!(
            "Usage: {} {PAIR_WORKER_ARG} <config> <parser> <source-file>",
            args[0]
        );
        return 2;
    }
    let all_configs = configs();
    let Some(config) = all_configs.iter().find(|config| config.name == args[2]) else {
        eprintln!("Unknown benchmark configuration: {}", args[2]);
        return 2;
    };
    let mut input = String::new();
    if let Err(error) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("Failed to read worker input: {error}");
        return 2;
    }
    match run_pair_worker(config, &args[3], &args[4], &input) {
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
    source_file: &str,
    input: &str,
    input_length: usize,
    token_count: usize,
) -> std::io::Result<BenchmarkResult> {
    let mut child = Command::new(std::env::current_exe()?)
        .args([PAIR_WORKER_ARG, config.name, parser_name, source_file])
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
        source_file,
    )
    .map_err(|error| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        std::io::Error::new(error.kind(), format!("{error}; worker stderr: {stderr}"))
    })
}

// ============================================================================
// Input Loading
// ============================================================================

/// Load all inputs for a config. Returns `(content, source_file)` pairs sorted by length.
///
/// - `TextLines`: each non-empty line in the listed files → `source_file` is `""`.
/// - `CodeFiles`: each file in the directory read whole → `source_file` is the filename.
fn load_inputs(source: &InputSource) -> Vec<(String, String)> {
    let mut inputs: Vec<(String, String)> = match source {
        InputSource::TextLines(paths) => {
            let mut lines = Vec::new();
            for path in *paths {
                let content = fs::read_to_string(path)
                    .unwrap_or_else(|_| panic!("Failed to read input file: {}", path));
                for line in content.lines() {
                    if !line.trim().is_empty() {
                        lines.push((line.to_string(), String::new()));
                    }
                }
            }
            lines
        }
        InputSource::CodeFiles(dir) => {
            let mut files = Vec::new();
            let entries =
                fs::read_dir(dir).unwrap_or_else(|_| panic!("Failed to read directory: {}", dir));
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name().unwrap().to_string_lossy().into_owned();
                    let content = fs::read_to_string(&path)
                        .unwrap_or_else(|_| panic!("Failed to read file: {:?}", path));
                    let content = content.trim().to_string();
                    files.push((content, filename));
                }
            }
            files
        }
    };

    // Sort by content length so the CSV grows monotonically (nicer for plotting)
    inputs.sort_by_key(|(content, _)| content.len());
    inputs
}

// ============================================================================
// Main Benchmark Logic
// ============================================================================

fn run_benchmarks(config: &GrammarConfig) -> std::io::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Benchmarking: {} grammar", config.name);
    println!("  Grammar: {}", config.grammar_path);
    match &config.input_source {
        InputSource::TextLines(paths) => println!("  Inputs (lines): {:?}", paths),
        InputSource::CodeFiles(dir) => println!("  Inputs (files in dir): {}", dir),
    }
    println!("{}", "=".repeat(60));

    // Setup CSV
    fs::create_dir_all(RESULT_DIR)?;
    let filename = output_path(config.name);
    let mut csv_file = File::create(&filename)?;
    writeln!(
        csv_file,
        "parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,recognized,parse_correct,status,source_file"
    )?;

    // Load grammar
    let grammar =
        grammars::load_grammar_from_file(config.grammar_path).expect("Failed to load grammar");
    println!("✓ Grammar loaded.");

    // Generate or reuse GLR tables
    if config.generate_table {
        println!("✓ Generating GLR table...");
        fs::create_dir_all("table")?;
        let table_gen = table_generator::TableGenerator::new(&grammar);
        table_gen
            .export_to_csv_numeric(config.table_path)
            .expect("Failed to export GLR table");
        println!("✓ GLR table generated.");
    } else {
        println!("✓ Skipping table generation (using existing tables)...");
    }

    println!("\n✓ Writing results to: {}", filename);

    // Load inputs
    let inputs = load_inputs(&config.input_source);
    println!(
        "Found {} inputs (lengths: {} to {} bytes)",
        inputs.len(),
        inputs.first().map(|(c, _)| c.len()).unwrap_or(0),
        inputs.last().map(|(c, _)| c.len()).unwrap_or(0),
    );

    let total_inputs = inputs.len();

    for (idx, (content, source_file)) in inputs.iter().enumerate() {
        let input_len = content.len();

        // Tokenize
        let tokens = match grammar.tokenize(content) {
            Some(t) => t,
            None => {
                let label = if source_file.is_empty() {
                    format!("#{}", idx + 1)
                } else {
                    source_file.clone()
                };
                eprintln!("\n[FATAL] Tokenization failed for '{}' — stopping.", label);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Tokenize failure: file={}", label),
                ));
            }
        };
        let token_count = tokens.len();
        // Display label: filename for CodeFiles, index for TextLines
        let display = if source_file.is_empty() {
            format!("#{}", idx + 1)
        } else {
            source_file.clone()
        };

        // Show a preview (replace newlines for readability)
        let preview: String = content
            .chars()
            .take(60)
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
            .collect();
        println!(
            "\n  [{}/{}] [{}] {} bytes, {} tokens: {}",
            idx + 1,
            total_inputs,
            display,
            input_len,
            token_count,
            preview.trim(),
        );

        for parser_name in config.parsers {
            let result = run_pair_in_worker(
                config,
                parser_name,
                source_file,
                content,
                input_len,
                token_count,
            )?;

            if result.status == "TIMEOUT" {
                println!(
                    "    [TIMEOUT] {:8}: invocation exceeded {:?}",
                    result.parser, PARSE_TIMEOUT
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            if result.status == "PARSE_FAIL" {
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                return Err(std::io::Error::other(format!(
                    "Parse failure: parser={}, file={}",
                    result.parser, display
                )));
            }

            let r = if result.recognized { "R" } else { "✗" };
            let p = if result.parse_correct { "P" } else { "✗" };
            println!(
                "    [{}|{}] {:8}: {:>12.0} ns ± {:>8.0} ns ({} iters)",
                r, p, result.parser, result.median_time_ns, result.mad_ns, result.iterations
            );

            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?;
        }
    }

    Ok(())
}

fn run_main() {
    println!("Parser Benchmark Tool — Code File Mode");
    println!("=======================================");
    println!("Inputs: whole source files | Parsers: Leo, GLL, RNGLR, BRNGLR, LL, LR\n");
    println!("Configuration:");
    println!("  Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("  Timed iterations:  {}", TIMED_ITERATIONS);
    println!("  Per-invocation timeout: {:?}", PARSE_TIMEOUT);

    for config in configs() {
        if let Err(e) = run_benchmarks(&config) {
            eprintln!("Error running benchmarks for {}: {}", config.name, e);
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

    const TIMEOUT_CHILD_ENV: &str = "BENCHMARK_CODE_TIMEOUT_CHILD";

    #[test]
    fn output_path_uses_benchmark_code_result_folder() {
        assert_eq!(
            output_path("java_ws"),
            "results/benchmark_code/benchmark_java_ws.csv"
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
        let result = worker_result_from_output(Some(124), b"", "GLL", 42, 17, "A.java").unwrap();
        assert_eq!(result.parser, "GLL");
        assert_eq!(result.source_file, "A.java");
        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, "TIMEOUT");
    }

    #[test]
    fn parse_failure_still_runs_ten_timed_iterations() {
        let mut calls = 0;
        let result = benchmark_parser("Test", 4, 2, "input.txt", "test", || {
            calls += 1;
            None
        });
        assert_eq!(calls, 11);
        assert_eq!(result.iterations, 10);
        assert_eq!(result.status, "PARSE_FAIL");
    }
}
