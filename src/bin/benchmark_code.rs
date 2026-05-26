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
//!   Creates CSV files in results/ with columns:
//!   parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,
//!   iterations,recognized,parse_correct,status,source_file
use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{earley_leo, gll, glr};
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
const MIN_ITERATIONS: u32 = 10;
const MAX_ITERATIONS: u32 = 20;
const TARGET_TIME: Duration = Duration::from_millis(500);
const TIMEOUT_THRESHOLD: f64 = 1_000_000_000.0; // 1 second in ns

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

    fn conflict(parser_name: &str, input_length: usize, token_count: usize, source_file: &str) -> Self {
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
            source_file: source_file.to_string(),
        }
    }

    fn timeout(parser_name: &str, input_length: usize, token_count: usize, source_file: &str) -> Self {
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
}

// ============================================================================
// Measurement Functions
// ============================================================================

fn measure_peak_memory<F>(mut parse_fn: F) -> usize
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

    let _ = black_box(parse_fn());

    stop_signal.store(true, Ordering::Relaxed);
    let _ = sampler.join();

    let peak = peak_mem.load(Ordering::Relaxed);
    if peak > start_mem { peak - start_mem } else { 0 }
}

fn measure<F>(mut parse_fn: F) -> (f64, f64, u32)
where
    F: FnMut() -> Option<ParseTree>,
{
    for _ in 0..WARMUP_ITERATIONS {
        let _ = black_box(parse_fn());
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
        times.push(start.elapsed().as_nanos() as f64);
        iterations += 1;
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

    (median, mad, iterations)
}

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
            let entries = fs::read_dir(dir)
                .unwrap_or_else(|_| panic!("Failed to read directory: {}", dir));
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let filename = path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
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
    fs::create_dir_all("results")?;
    let filename = format!("results/benchmark_{}.csv", config.name);
    let mut csv_file = File::create(&filename)?;
    writeln!(
        csv_file,
        "parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,recognized,parse_correct,status,source_file"
    )?;

    // Load grammar
    let grammar = grammars::load_grammar_from_file(config.grammar_path)
        .expect("Failed to load grammar");
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

    // Instantiate parsers
    let mut rnglr = glr::RnglrParser::import_table_from_csv(config.table_path)
        .expect("Failed to load RNGLR table");
    let mut brnglr = glr::BrnglrParser::import_table_from_csv(config.table_path)
        .expect("Failed to load BRNGLR table");
    let mut gll_parser = gll::GLLParser::new(&grammar);
    let mut leo = earley_leo::LeoParser::new(grammar.clone());
    let conflict_parsers: HashSet<String> = HashSet::new();

    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    // Load inputs
    let inputs = load_inputs(&config.input_source);
    println!(
        "Found {} inputs (lengths: {} to {} bytes)",
        inputs.len(),
        inputs.first().map(|(c, _)| c.len()).unwrap_or(0),
        inputs.last().map(|(c, _)| c.len()).unwrap_or(0),
    );

    let mut failed_parsers: HashSet<String> = HashSet::new();
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
        let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

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
            // Skip conflicting parsers
            if conflict_parsers.contains(*parser_name) {
                let result = BenchmarkResult::conflict(parser_name, input_len, token_count, source_file);
                println!("    [CONFLICT] {:8}: grammar has conflicts, skipping", parser_name);
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            // Skip timed-out parsers
            if failed_parsers.contains(*parser_name) {
                let result = BenchmarkResult::timeout(parser_name, input_len, token_count, source_file);
                println!("    [TIMEOUT]  {:8}: previously timed out, skipping", parser_name);
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            // Helper macro: check recognition first; bail out immediately on failure.
            macro_rules! check_and_bail {
                ($parse_expr:expr) => {{
                    let (recognized, parse_correct) =
                        check_correctness(|| $parse_expr, content);
                    if !recognized {
                        let fail_result = BenchmarkResult {
                            parser: parser_name.to_string(),
                            input_length: input_len,
                            token_count,
                            median_time_ns: 0.0,
                            mad_ns: 0.0,
                            peak_memory_bytes: 0,
                            iterations: 0,
                            recognized: false,
                            parse_correct: false,
                            status: "PARSE_FAIL".to_string(),
                            source_file: source_file.clone(),
                        };
                        writeln!(csv_file, "{}", fail_result.to_csv_row())?;
                        csv_file.flush()?;
                        eprintln!(
                            "\n[FATAL] Parser '{}' failed to recognize '{}' — stopping.",
                            parser_name, display
                        );
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "Parse failure: parser={}, file={}",
                                parser_name, display
                            ),
                        ));
                    }
                    (recognized, parse_correct)
                }};
            }

            let result = match *parser_name {
                "Leo" => {
                    let (recognized, parse_correct) =
                        check_and_bail!(leo.parse(tokens.clone()));
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
                        source_file: source_file.clone(),
                    }
                }
                "GLL" => {
                    let (recognized, parse_correct) =
                        check_and_bail!(gll_parser.parse(&tokens));
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
                        source_file: source_file.clone(),
                    }
                }
                "RNGLR" => {
                    let (recognized, parse_correct) =
                        check_and_bail!(rnglr.parse(&glr_tokens));
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
                        source_file: source_file.clone(),
                    }
                }
                "BRNGLR" => {
                    let (recognized, parse_correct) =
                        check_and_bail!(brnglr.parse(&glr_tokens));
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
                        source_file: source_file.clone(),
                    }
                }
                _ => continue,
            };

            let r = if result.recognized { "R" } else { "✗" };
            let p = if result.parse_correct { "P" } else { "✗" };
            println!(
                "    [{}|{}] {:8}: {:>12.0} ns ± {:>8.0} ns ({} iters)",
                r, p, result.parser, result.median_time_ns, result.mad_ns, result.iterations
            );

            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?;

            if result.median_time_ns > TIMEOUT_THRESHOLD {
                println!(
                    "    [TIMEOUT] {} exceeded threshold, skipping larger inputs",
                    result.parser
                );
                failed_parsers.insert(result.parser);
            }
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
    println!("  Min iterations:    {}", MIN_ITERATIONS);
    println!("  Max iterations:    {}", MAX_ITERATIONS);
    println!("  Target time:       {:?}", TARGET_TIME);

    for config in configs() {
        if let Err(e) = run_benchmarks(&config) {
            eprintln!("Error running benchmarks for {}: {}", config.name, e);
        }
    }

    println!("\n✓ Benchmarking complete!");
}

fn main() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024) // 128 MB stack
        .spawn(run_main)
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}
