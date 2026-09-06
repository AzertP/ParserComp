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
//!   results/benchmark_lex/benchmark_<language>.csv
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
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Language configurations
// ============================================================================

#[allow(dead_code)]
const DEFAULT_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];
const EXCLUDED_SOURCE_FILES: &[&str] = &[
    // Uses C# 2.0 generic method declarations/invocations outside the adapted C# 1.2 grammar scope.
    "1941974.cs",
    "2302133.cs",
    // Uses C# 2.0 property accessor modifier syntax outside the adapted C# 1.2 grammar scope.
    "3217761.cs",
    "3217789.cs",
    // Uses C# 2.0 iterator/yield syntax outside the adapted C# 1.2 grammar scope.
    "3242471.cs",
    // Uses C# 3.0 object initializer syntax outside the adapted C# 1.2 grammar scope.
    "2467486.cs",
    "2696682.cs",
    "2892858.cs",
    "4399879.cs",
    "4407712.cs",
    "4408113.cs",
    "4414104.cs",
    // Uses C# 3.0 collection initializer syntax outside the adapted C# 1.2 grammar scope.
    "2440111.cs",
    "2442986.cs",
    "4774386.cs",
    // Uses C# 6.0 expression-bodied member syntax outside the adapted C# 1.2 grammar scope.
    "4312573.cs",
    "4313948.cs",
    "4313968.cs",
    "4314068.cs",
    // Uses nested generic type closers (`>>`), which require contextual lexer handling.
    "3232886.cs",
    "3303252.cs",
    "4231178.cs",
    "4774920.cs",
    // Uses tuple assignment syntax outside the adapted C# 1.2 grammar scope.
    "9518095.cs",
];

struct LangConfig {
    name: &'static str,
    lexer_spec: &'static str,
    grammar_path: &'static str,
    input_dir: &'static str,
    file_exts: &'static [&'static str],
    table_path: &'static str,
    generate_table: bool,
    parsers: &'static [&'static str],
}

const CONFIGS: &[LangConfig] = &[
    LangConfig {
        name: "csharp_tok",
        lexer_spec: "grammars/lexer/csharp_regex.json",
        grammar_path: "grammars/csharp_tok.json",
        input_dir: "input/code/Csharp",
        file_exts: &[".cs"],
        table_path: "table/csharp_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
    LangConfig {
        name: "c_tok",
        lexer_spec: "grammars/lexer/c_regex.json",
        grammar_path: "grammars/ansi_c_tok.json",
        input_dir: "input/code/C",
        file_exts: &[".c"],
        table_path: "table/ansi_c_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
    // Uncomment when lexer specs and tokenised grammars are ready:
    LangConfig {
        name: "cpp_tok",
        lexer_spec: "grammars/lexer/cpp_regex.json",
        grammar_path: "grammars/cpp_tok.json",
        input_dir: "input/code/C++",
        file_exts: &[".cpp"],
        table_path: "table/cpp_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
    LangConfig {
        name: "java_tok",
        lexer_spec: "grammars/lexer/java_regex.json",
        grammar_path: "grammars/java_tok.json",
        input_dir: "input/code/Java",
        file_exts: &[".java"],
        table_path: "table/java_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
    LangConfig {
        name: "pascal_tok",
        lexer_spec: "grammars/lexer/pascal_regex.json",
        grammar_path: "grammars/pascal_tok.json",
        input_dir: "input/code/Pascal ISO",
        file_exts: &[".pas"],
        table_path: "table/pascal_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
];

// ============================================================================
// Timing constants — identical to benchmark_csv.rs
// ============================================================================

const WARMUP_ITERATIONS: u32 = 1;
const TIMED_ITERATIONS: u32 = 10;
const PARSE_TIMEOUT: Duration = Duration::from_secs(1);
const TIMEOUT_EXIT_CODE: i32 = 124;
const PAIR_WORKER_ARG: &str = "--pair-worker";
const WORKER_RESULT_PREFIX: &str = "WORKER_RESULT\t";
const RESULT_DIR: &str = "results/benchmark_lex";

fn output_path(config_name: &str) -> String {
    format!(
        "{}/benchmark_{}.csv",
        RESULT_DIR,
        config_name.replace(['+', ' '], "_")
    )
}

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

    fn from_csv_row(row: &str) -> Result<Self, String> {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() != 14 {
            return Err(format!(
                "expected 14 worker result fields, found {}",
                fields.len()
            ));
        }
        Ok(BenchmarkResult {
            language: fields[0].to_string(),
            size_category: fields[1].to_string(),
            file: fields[2].to_string(),
            bytes: fields[3]
                .parse()
                .map_err(|e| format!("invalid byte count: {e}"))?,
            parser: fields[4].to_string(),
            input_length: fields[5]
                .parse()
                .map_err(|e| format!("invalid input length: {e}"))?,
            token_count: fields[6]
                .parse()
                .map_err(|e| format!("invalid token count: {e}"))?,
            median_time_ns: fields[7]
                .parse()
                .map_err(|e| format!("invalid median time: {e}"))?,
            mad_ns: fields[8].parse().map_err(|e| format!("invalid MAD: {e}"))?,
            peak_memory_bytes: fields[9]
                .parse()
                .map_err(|e| format!("invalid peak memory: {e}"))?,
            iterations: fields[10]
                .parse()
                .map_err(|e| format!("invalid iteration count: {e}"))?,
            recognized: fields[11]
                .parse()
                .map_err(|e| format!("invalid recognized flag: {e}"))?,
            parse_correct: fields[12]
                .parse()
                .map_err(|e| format!("invalid parse-correct flag: {e}"))?,
            status: fields[13].to_string(),
        })
    }
}

fn worker_result_from_output(
    exit_code: Option<i32>,
    stdout: &[u8],
    cfg: &LangConfig,
    input: &InputCase,
    parser_name: &str,
) -> std::io::Result<BenchmarkResult> {
    if exit_code == Some(TIMEOUT_EXIT_CODE) {
        return Ok(BenchmarkResult::timeout(
            cfg.name,
            &input.size_category,
            &input.file,
            input.bytes,
            parser_name,
            input.token_count,
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
// Measurement functions — same as benchmark_csv.rs
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
    let ((warmup_result, peak_mem), _) = run_with_hard_timeout(PARSE_TIMEOUT, || {
        measure_peak_memory_with_result(&mut parse_fn)
    });
    let (median, mad, iters) = measure(&mut parse_fn);
    if warmup_result.is_none() {
        let mut result =
            BenchmarkResult::parse_fail(lang, size, file, bytes, parser_name, token_count);
        result.median_time_ns = median;
        result.mad_ns = mad;
        result.peak_memory_bytes = peak_mem;
        result.iterations = iters;
        return result;
    }
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

fn collect_files(dir: &str, exts: &[&str]) -> Vec<PathBuf> {
    let bare_exts: Vec<String> = exts
        .iter()
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .collect();
    let mut files: Vec<PathBuf> = fs::read_dir(Path::new(dir))
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let p = e.ok()?.path();
            if has_configured_extension(&p, &bare_exts) && !is_excluded_source_file(&p) {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

fn has_configured_extension(path: &Path, bare_exts: &[String]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            bare_exts
                .iter()
                .any(|wanted| ext.eq_ignore_ascii_case(wanted))
        })
        .unwrap_or(false)
}

fn is_excluded_source_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| EXCLUDED_SOURCE_FILES.contains(&name))
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

fn run_pair_worker(
    cfg: &LangConfig,
    parser_name: &str,
    file: &str,
) -> Result<BenchmarkResult, String> {
    let lexer =
        Lexer::from_file(cfg.lexer_spec).map_err(|e| format!("failed to load lexer: {e}"))?;
    let grammar = grammars::load_grammar_from_file(cfg.grammar_path)
        .map_err(|e| format!("failed to load grammar: {e}"))?;
    let path = Path::new(cfg.input_dir).join(file);
    let mut inputs = load_inputs(&[path], &lexer, &grammar);
    let input = inputs
        .pop()
        .ok_or_else(|| format!("failed to load worker input {file}"))?;

    match parser_name {
        "Leo" => {
            let mut parser = earley_leo::LeoParser::new(grammar.clone());
            Ok(benchmark_parser(
                cfg.name,
                &input.size_category,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                || parser.parse(input.ids.clone()),
            ))
        }
        "GLL" => {
            let mut parser = gll::GLLParser::new(&grammar);
            Ok(benchmark_parser(
                cfg.name,
                &input.size_category,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                || parser.parse_one(&input.ids),
            ))
        }
        "RNGLR" => {
            let mut parser = glr::RnglrParser::import_table_from_csv(cfg.table_path)
                .map_err(|e| format!("failed to load RNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            Ok(benchmark_parser(
                cfg.name,
                &input.size_category,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                || parser.parse(&input.glr_ids),
            ))
        }
        "BRNGLR" => {
            let mut parser = glr::BrnglrParser::import_table_from_csv(cfg.table_path)
                .map_err(|e| format!("failed to load BRNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            Ok(benchmark_parser(
                cfg.name,
                &input.size_category,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                || parser.parse(&input.glr_ids),
            ))
        }
        _ => Err(format!("unknown parser: {parser_name}")),
    }
}

fn pair_worker_main(args: &[String]) -> i32 {
    if args.len() != 5 {
        eprintln!(
            "Usage: {} {PAIR_WORKER_ARG} <config> <parser> <file>",
            args[0]
        );
        return 2;
    }
    let Some(cfg) = CONFIGS.iter().find(|cfg| cfg.name == args[2]) else {
        eprintln!("Unknown benchmark configuration: {}", args[2]);
        return 2;
    };
    match run_pair_worker(cfg, &args[3], &args[4]) {
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
    cfg: &LangConfig,
    input: &InputCase,
    parser_name: &str,
) -> std::io::Result<BenchmarkResult> {
    let output = Command::new(std::env::current_exe()?)
        .args([PAIR_WORKER_ARG, cfg.name, parser_name, &input.file])
        .output()?;
    worker_result_from_output(
        output.status.code(),
        &output.stdout,
        cfg,
        input,
        parser_name,
    )
    .map_err(|error| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        std::io::Error::new(error.kind(), format!("{error}; worker stderr: {stderr}"))
    })
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

    // Discover source files
    let files = collect_files(cfg.input_dir, cfg.file_exts);
    if files.is_empty() {
        eprintln!(
            "[SKIP] No {:?} files found in {}.",
            cfg.file_exts, cfg.input_dir
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
    fs::create_dir_all(RESULT_DIR)?;
    let out_path = output_path(cfg.name);
    let mut csv_file = File::create(&out_path)?;
    writeln!(
        csv_file,
        "language,size_category,file,bytes,\
         parser,input_length,token_count,median_time_ns,mad_ns,\
         peak_memory_bytes,iterations,recognized,parse_correct,status"
    )?;
    println!("\n✓ Writing results to: {}", out_path);

    for (idx, input) in inputs.iter().enumerate() {
        let display_file = &input.file[..input.file.len().min(60)];
        println!(
            "\n  Input #{} [{}] {} bytes, {} tokens",
            idx + 1,
            display_file,
            input.bytes,
            input.token_count
        );

        for &parser_name in cfg.parsers {
            let result = run_pair_in_worker(cfg, input, parser_name)?;

            if result.status == "TIMEOUT" {
                println!(
                    "    [TIMEOUT] {:8}: invocation exceeded {:?}",
                    result.parser, PARSE_TIMEOUT
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

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
    println!("  Timed iterations  : {}", TIMED_ITERATIONS);
    println!("  Per-invocation timeout: {:?}", PARSE_TIMEOUT);

    for cfg in CONFIGS {
        if let Err(e) = run_config(cfg) {
            eprintln!("[ERROR] {}: {}", cfg.name, e);
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

    // Use 128 MB stack to handle deep recursion in parsers (same as benchmark_csv.rs)
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(run_main)
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT_CHILD_ENV: &str = "BENCHMARK_LEX_TIMEOUT_CHILD";

    #[test]
    fn output_path_uses_benchmark_lex_result_folder() {
        assert_eq!(
            output_path("java_tok"),
            "results/benchmark_lex/benchmark_java_tok.csv"
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
        let input = InputCase {
            file: "A.java".to_string(),
            size_category: "S".to_string(),
            bytes: 42,
            ids: vec![],
            glr_ids: vec![],
            token_count: 17,
        };
        let result = worker_result_from_output(Some(124), b"", &CONFIGS[0], &input, "GLL").unwrap();
        assert_eq!(result.parser, "GLL");
        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, "TIMEOUT");
    }

    #[test]
    fn parse_failure_still_runs_ten_timed_iterations() {
        let mut calls = 0;
        let result = benchmark_parser("test", "S", "input", 4, "Test", 2, || {
            calls += 1;
            None
        });
        assert_eq!(calls, 11);
        assert_eq!(result.iterations, 10);
        assert_eq!(result.status, "PARSE_FAIL");
    }

    #[test]
    fn collect_files_omits_configured_unsupported_sources() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "parser_comparison_benchmark_lex_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).expect("create temp test directory");
        for excluded in [
            "9518095.cs",
            "1941974.cs",
            "2302133.cs",
            "2696682.cs",
            "4774920.cs",
            "3303252.cs",
            "3232886.cs",
            "4231178.cs",
            "2442986.cs",
            "4774386.cs",
            "2440111.cs",
            "2467486.cs",
            "3242471.cs",
            "4399879.cs",
            "2892858.cs",
            "4312573.cs",
            "4313948.cs",
            "4313968.cs",
            "4314068.cs",
            "4407712.cs",
            "4408113.cs",
            "4414104.cs",
            "3217761.cs",
            "3217789.cs",
        ] {
            fs::write(tmp_dir.join(excluded), "class C {}").expect("write excluded source");
        }
        fs::write(tmp_dir.join("2433596.cs"), "class C {}").expect("write included source");

        let files = collect_files(tmp_dir.to_str().unwrap(), &[".cs"]);
        let names: Vec<_> = files
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .collect();

        assert_eq!(names, vec!["2433596.cs"]);

        fs::remove_dir_all(&tmp_dir).expect("remove temp test directory");
    }

    #[test]
    fn collect_files_accepts_multiple_extensions_case_insensitively() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "parser_comparison_benchmark_lex_ext_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).expect("create temp test directory");
        for file in ["a.foo", "b.BAR", "c.baz", "ignored.txt"] {
            fs::write(tmp_dir.join(file), "sample").expect("write source");
        }

        let files = collect_files(tmp_dir.to_str().unwrap(), &[".foo", ".bar", ".baz"]);
        let names: Vec<_> = files
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .collect();

        assert_eq!(names, vec!["a.foo", "b.BAR", "c.baz"]);

        fs::remove_dir_all(&tmp_dir).expect("remove temp test directory");
    }
}
