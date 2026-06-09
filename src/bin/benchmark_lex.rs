//! Parser benchmark on real source files (lex → encode → parse pipeline).
//!
//! For each language, discovers source files in `input/code/<Language>/`,
//! lexes each one, and benchmarks Leo, GLL, RNGLR, and BRNGLR. RNGLR and
//! BRNGLR use a CSV-backed GLR/LR table, matching the file-I/O path used by
//! benchmark_csv.rs and benchmark_code.rs.
//!
//! Closely follows the structure and conventions of benchmark_csv.rs.
//!
//! Usage:
//!   cargo run --release --bin benchmark_lex
//!
//! Output:
//!   results/benchmark_lex_<language>.csv
//!   CSV columns:
//!     language, size_category, file, bytes,
//!     parser, token_count, median_time_ns, mad_ns,
//!     peak_memory_bytes, iterations, recognized, parse_correct, status

use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{earley_leo, gll, glr};
use std::collections::HashSet;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Language configurations
// ============================================================================

const DEFAULT_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];

struct LangConfig {
    name: &'static str,
    lexer_spec: &'static str,
    grammar_path: &'static str,
    input_dir: &'static str,
    file_ext: &'static str,
    table_path: &'static str,
    generate_table: bool,
    parsers: &'static [&'static str],
}

const CONFIGS: &[LangConfig] = &[
    LangConfig {
        name: "c_tok",
        lexer_spec: "grammars/lexer/c_regex.json",
        grammar_path: "grammars/ansi_c_tok.json",
        input_dir: "input/code/C",
        file_ext: ".c",
        table_path: "table/ansi_c_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
    // Uncomment when lexer specs and tokenised grammars are ready:
    // LangConfig {
    //     name: "C++",
    //     lexer_spec: "grammars/lexer/cpp_regex.json",
    //     grammar_path: "grammars/cpp_tok.json",
    //     input_dir: "input/code/C++",
    //     file_ext: ".cpp",
    //     table_path: "table/cpp_tok_glr_table.csv",
    //     generate_table: true,
    //     parsers: DEFAULT_PARSERS,
    // },
    // LangConfig {
    //     name: "Java",
    //     lexer_spec: "grammars/lexer/java_regex.json",
    //     grammar_path: "grammars/jsl18_tok.json",
    //     input_dir: "input/code/Java",
    //     file_ext: ".java",
    //     table_path: "table/java_tok_glr_table.csv",
    //     generate_table: true,
    //     parsers: DEFAULT_PARSERS,
    // },
    // LangConfig {
    //     name: "FreePascal",
    //     lexer_spec: "grammars/lexer/pascal_regex.json",
    //     grammar_path: "grammars/pascal_tok.json",
    //     input_dir: "input/code/FreePascal",
    //     file_ext: ".pas",
    //     table_path: "table/pascal_tok_glr_table.csv",
    //     generate_table: true,
    //     parsers: DEFAULT_PARSERS,
    // },
];

// ============================================================================
// Timing constants — identical to benchmark_csv.rs
// ============================================================================

const WARMUP_ITERATIONS: u32 = 1;
const MIN_ITERATIONS: u32 = 10;
const MAX_ITERATIONS: u32 = 20;
const TARGET_TIME: Duration = Duration::from_millis(500);
const TIMEOUT_THRESHOLD: f64 = 1_000_000_000.0; // 1 second in ns

// ============================================================================
// BenchmarkResult — mirrors benchmark_csv.rs, with extra leading columns
// ============================================================================

#[derive(Clone)]
struct BenchmarkResult {
    // Extra columns for code-file benchmarks
    language: String,
    size_category: String,
    file: String,
    bytes: usize,
    // Same fields as benchmark_csv.rs
    parser: String,
    input_length: usize, // = token_count
    token_count: usize,
    median_time_ns: f64,
    mad_ns: f64,
    peak_memory_bytes: usize,
    iterations: u32,
    recognized: bool,
    parse_correct: bool,
    status: String,
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.2},{:.2},{},{},{},{},{}",
            self.language,
            self.size_category,
            self.file,
            self.bytes,
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
        )
    }

    fn timeout(
        lang: &str,
        size: &str,
        file: &str,
        bytes: usize,
        parser_name: &str,
        token_count: usize,
    ) -> Self {
        BenchmarkResult {
            language: lang.to_string(),
            size_category: size.to_string(),
            file: file.to_string(),
            bytes,
            parser: parser_name.to_string(),
            input_length: token_count,
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

    fn parse_fail(
        lang: &str,
        size: &str,
        file: &str,
        bytes: usize,
        parser_name: &str,
        token_count: usize,
    ) -> Self {
        BenchmarkResult {
            language: lang.to_string(),
            size_category: size.to_string(),
            file: file.to_string(),
            bytes,
            parser: parser_name.to_string(),
            input_length: token_count,
            token_count,
            median_time_ns: 0.0,
            mad_ns: 0.0,
            peak_memory_bytes: 0,
            iterations: 0,
            recognized: false,
            parse_correct: false,
            status: "PARSE_FAIL".to_string(),
        }
    }
}

// ============================================================================
// Measurement functions — same as benchmark_csv.rs
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
    if peak > start_mem {
        peak - start_mem
    } else {
        0
    }
}

fn measure<F>(mut parse_fn: F) -> (f64, f64, u32)
where
    F: FnMut() -> Option<ParseTree>,
{
    for _ in 0..WARMUP_ITERATIONS {
        let _ = black_box(parse_fn());
    }

    let mut times: Vec<f64> = Vec::new();
    let start_measure = Instant::now();

    loop {
        if times.len() as u32 >= MAX_ITERATIONS {
            break;
        }
        if times.len() as u32 >= MIN_ITERATIONS && start_measure.elapsed() >= TARGET_TIME {
            break;
        }
        let start = Instant::now();
        let _ = black_box(parse_fn());
        times.push(start.elapsed().as_nanos() as f64);
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
    let mut devs: Vec<f64> = times.iter().map(|t| (t - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if devs.len() % 2 == 0 {
        (devs[devs.len() / 2 - 1] + devs[devs.len() / 2]) / 2.0
    } else {
        devs[devs.len() / 2]
    };

    (median, mad, times.len() as u32)
}

fn benchmark_parser<F>(
    lang: &str,
    size: &str,
    file: &str,
    bytes: usize,
    parser_name: &str,
    token_count: usize,
    mut parse_fn: F,
) -> BenchmarkResult
where
    F: FnMut() -> Option<ParseTree>,
{
    if parse_fn().is_none() {
        return BenchmarkResult::parse_fail(lang, size, file, bytes, parser_name, token_count);
    }

    let peak_mem = measure_peak_memory(&mut parse_fn);
    let (median, mad, iters) = measure(&mut parse_fn);
    BenchmarkResult {
        language: lang.to_string(),
        size_category: size.to_string(),
        file: file.to_string(),
        bytes,
        parser: parser_name.to_string(),
        input_length: token_count,
        token_count,
        median_time_ns: median,
        mad_ns: mad,
        peak_memory_bytes: peak_mem,
        iterations: iters,
        recognized: true,
        parse_correct: true,
        status: "OK".to_string(),
    }
}

fn collect_files(dir: &str, ext: &str) -> Vec<PathBuf> {
    let bare_ext = ext.trim_start_matches('.');
    let mut files: Vec<PathBuf> = fs::read_dir(Path::new(dir))
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension().and_then(|s| s.to_str()) == Some(bare_ext) {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

struct InputCase {
    file: String,
    size_category: String,
    bytes: usize,
    ids: Vec<u32>,
    glr_ids: Vec<i32>,
    token_count: usize,
}

fn load_inputs(
    files: &[PathBuf],
    lexer: &Lexer,
    grammar: &grammars::NumericGrammar,
) -> Vec<InputCase> {
    let mut inputs = Vec::new();

    for path in files {
        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let size_category = size_prefix(&file).to_string();
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[WARN] Cannot read {:?}: {}", path, e);
                continue;
            }
        };
        let bytes = source.len();

        let toks = match lexer.lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[WARN] Lex error {}: {}", file, e);
                continue;
            }
        };
        let ids: Vec<u32> = match encode(&toks, grammar) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[WARN] Encode error {}: {}", file, e);
                continue;
            }
        };
        let glr_ids: Vec<i32> = ids.iter().map(|&t| (t + 1) as i32).collect();
        let token_count = ids.len();

        inputs.push(InputCase {
            file,
            size_category,
            bytes,
            ids,
            glr_ids,
            token_count,
        });
    }

    inputs.sort_by(|a, b| {
        a.token_count
            .cmp(&b.token_count)
            .then_with(|| a.bytes.cmp(&b.bytes))
            .then_with(|| a.file.cmp(&b.file))
    });
    inputs
}

fn size_prefix(name: &str) -> &str {
    for cat in &["XS", "S", "M", "L", "XL"] {
        if name.starts_with(cat) && name[cat.len()..].starts_with('_') {
            return cat;
        }
    }
    "?"
}

// ============================================================================
// Per-language benchmark — mirrors run_benchmarks() in benchmark_csv.rs
// ============================================================================

fn run_config(cfg: &LangConfig) -> std::io::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Benchmarking: {} (tokenized grammar)", cfg.name);
    println!("  Grammar : {}", cfg.grammar_path);
    println!("  Lexer   : {}", cfg.lexer_spec);
    println!("  Input   : {}", cfg.input_dir);
    println!("  Table   : {}", cfg.table_path);
    println!("{}", "=".repeat(60));

    // Load lexer + grammar
    let lexer = Lexer::from_file(cfg.lexer_spec)
        .unwrap_or_else(|e| panic!("Cannot load lexer spec {}: {}", cfg.lexer_spec, e));
    let grammar = grammars::load_grammar_from_file(cfg.grammar_path)
        .unwrap_or_else(|e| panic!("Cannot load grammar {}: {}", cfg.grammar_path, e));

    println!(
        "✓ Grammar loaded ({} terminals, {} non-terminals).",
        grammar.num_terminals(),
        grammar.num_non_terminals()
    );

    if cfg.generate_table {
        println!("✓ Generating GLR table...");
        fs::create_dir_all("table")?;
        let table_gen = table_generator::TableGenerator::new(&grammar);
        table_gen
            .export_to_csv_numeric(cfg.table_path)
            .expect("Failed to export GLR table");
        println!("✓ GLR table generated.");
    } else {
        println!("✓ Skipping table generation (using existing table)...");
    }

    // Instantiate parsers. RNGLR/BRNGLR come from the CSV-backed table.
    let mut leo = earley_leo::LeoParser::new(grammar.clone());
    let mut gll_parser = gll::GLLParser::new(&grammar);
    let mut rnglr = glr::RnglrParser::import_table_from_csv(cfg.table_path)
        .expect("Failed to load RNGLR table");
    let mut brnglr = glr::BrnglrParser::import_table_from_csv(cfg.table_path)
        .expect("Failed to load BRNGLR table");
    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    // Discover source files
    let files = collect_files(cfg.input_dir, cfg.file_ext);
    if files.is_empty() {
        eprintln!(
            "[SKIP] No {} files found in {}.",
            cfg.file_ext, cfg.input_dir
        );
        return Ok(());
    }
    println!("✓ {} source files found.", files.len());

    let inputs = load_inputs(&files, &lexer, &grammar);
    if inputs.is_empty() {
        eprintln!("[SKIP] No benchmarkable inputs found for {}.", cfg.name);
        return Ok(());
    }
    println!(
        "✓ {} benchmarkable inputs ({} to {} tokens).",
        inputs.len(),
        inputs.first().map(|input| input.token_count).unwrap_or(0),
        inputs.last().map(|input| input.token_count).unwrap_or(0)
    );

    // Open output CSV
    fs::create_dir_all("results")?;
    let out_path = format!(
        "results/benchmark_{}.csv",
        cfg.name.replace(['+', ' '], "_")
    );
    let mut csv_file = File::create(&out_path)?;
    writeln!(
        csv_file,
        "language,size_category,file,bytes,\
         parser,input_length,token_count,median_time_ns,mad_ns,\
         peak_memory_bytes,iterations,recognized,parse_correct,status"
    )?;
    println!("\n✓ Writing results to: {}", out_path);

    let mut failed_parsers: HashSet<String> = HashSet::new();

    for (idx, input) in inputs.iter().enumerate() {
        println!(
            "\n  Input #{} [{:>3}] {} bytes, {} tokens",
            idx + 1,
            input.size_category,
            input.bytes,
            input.token_count
        );
        println!("  {}", &input.file[..input.file.len().min(60)]);

        for &parser_name in cfg.parsers {
            // Skip previously timed-out parsers
            if failed_parsers.contains(parser_name) {
                let result = BenchmarkResult::timeout(
                    cfg.name,
                    &input.size_category,
                    &input.file,
                    input.bytes,
                    parser_name,
                    input.token_count,
                );
                println!(
                    "    [TIMEOUT]  {:8}: previously timed out, skipping",
                    parser_name
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            let result = match parser_name {
                "Leo" => benchmark_parser(
                    cfg.name,
                    &input.size_category,
                    &input.file,
                    input.bytes,
                    "Leo",
                    input.token_count,
                    || leo.parse(input.ids.clone()),
                ),
                "GLL" => benchmark_parser(
                    cfg.name,
                    &input.size_category,
                    &input.file,
                    input.bytes,
                    "GLL",
                    input.token_count,
                    || gll_parser.parse(&input.ids),
                ),
                "RNGLR" => benchmark_parser(
                    cfg.name,
                    &input.size_category,
                    &input.file,
                    input.bytes,
                    "RNGLR",
                    input.token_count,
                    || rnglr.parse(&input.glr_ids),
                ),
                "BRNGLR" => benchmark_parser(
                    cfg.name,
                    &input.size_category,
                    &input.file,
                    input.bytes,
                    "BRNGLR",
                    input.token_count,
                    || brnglr.parse(&input.glr_ids),
                ),
                _ => continue,
            };

            let recog_sym = if result.recognized { "R" } else { "✗" };
            let parse_sym = if result.parse_correct { "P" } else { "✗" };
            println!(
                "    [{}|{}] {:8}: {:>12.0} ns ± {:>8.0} ns  ({} iters)",
                recog_sym,
                parse_sym,
                result.parser,
                result.median_time_ns,
                result.mad_ns,
                result.iterations
            );

            if !result.recognized {
                eprintln!(
                    "    [STOP] {} rejected {} ({} tokens); stopping {} benchmark.",
                    result.parser, input.file, input.token_count, cfg.name
                );
            }

            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?;

            if !result.recognized {
                return Ok(());
            }

            if result.median_time_ns > TIMEOUT_THRESHOLD {
                println!(
                    "    [TIMEOUT] {} exceeded {}s threshold — skipping subsequent files",
                    result.parser,
                    TIMEOUT_THRESHOLD as u64 / 1_000_000_000
                );
                failed_parsers.insert(result.parser.clone());
            }
        }
    }

    println!("\n✓ Written to {}", out_path);
    Ok(())
}

// ============================================================================
// Main — same pattern as benchmark_csv.rs (large stack thread)
// ============================================================================

fn run_main() {
    println!("Lexer + Parser Benchmark Tool");
    println!("=============================");
    println!(
        "Benchmarks Leo, GLL, RNGLR, and BRNGLR on real source files from input/code/<Language>/\n"
    );
    println!("Configuration:");
    println!("  Warmup iterations : {}", WARMUP_ITERATIONS);
    println!("  Min iterations    : {}", MIN_ITERATIONS);
    println!("  Max iterations    : {}", MAX_ITERATIONS);
    println!("  Target time       : {:?}", TARGET_TIME);

    for cfg in CONFIGS {
        if let Err(e) = run_config(cfg) {
            eprintln!("[ERROR] {}: {}", cfg.name, e);
        }
    }

    println!("\n✓ Benchmarking complete!");
}

fn main() {
    // Use 128 MB stack to handle deep recursion in parsers (same as benchmark_csv.rs)
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(run_main)
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}
