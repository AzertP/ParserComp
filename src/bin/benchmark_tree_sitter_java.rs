//! Tree-sitter Java baseline plus local parser comparison for real Java sources.
//!
//! This measures tree-sitter-java over the same `input/code/Java/*.java`
//! corpus used by `benchmark_lex.rs`, then runs Leo, GLL, RNGLR, and BRNGLR
//! over the tree-sitter-derived `java_tree` token grammar.
//!
//! Usage:
//!   cargo run --release --bin benchmark_tree_sitter_java
//!   cargo run --release --bin benchmark_tree_sitter_java -- --generate-table-only
//!   cargo run --release --bin benchmark_tree_sitter_java -- --regenerate-table
//!
//! Output:
//!   results/benchmark_tree_sitter_java/benchmark_tree_sitter_java.csv

use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{earley_leo, gll, glr};
use std::env;
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
use tree_sitter::{Node, Parser};

const INPUT_DIR: &str = "input/code/Java";
const JAVA_TREE_LEXER_SPEC: &str = "grammars/lexer/java_tree_regex.json";
const JAVA_TREE_GRAMMAR: &str = "grammars/java_tree.json";
const JAVA_TREE_GLR_TABLE: &str = "table/java_tree_glr_table.csv";
const RESULT_DIR: &str = "results/benchmark_tree_sitter_java";
const OUT_PATH: &str = "results/benchmark_tree_sitter_java/benchmark_tree_sitter_java.csv";
const LOCAL_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];

const WARMUP_ITERATIONS: u32 = 1;
const TIMED_ITERATIONS: u32 = 10;
const PARSE_TIMEOUT: Duration = Duration::from_secs(1);
const TIMEOUT_EXIT_CODE: i32 = 124;
const PAIR_WORKER_ARG: &str = "--pair-worker";
const WORKER_RESULT_PREFIX: &str = "WORKER_RESULT\t";

struct RunOptions {
    regenerate_table: bool,
    generate_table_only: bool,
}

#[derive(Clone)]
struct InputCase {
    file: String,
    size_category: String,
    source: String,
    bytes: usize,
    ids: Vec<u32>,
    glr_ids: Vec<i32>,
    token_count: usize,
}

#[derive(Clone)]
struct BenchmarkResult {
    language: String,
    size_category: String,
    file: String,
    bytes: usize,
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
    error_nodes: usize,
    missing_nodes: usize,
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

    fn parse_fail(input: &InputCase, parser_name: &str) -> Self {
        BenchmarkResult {
            language: "java_tree".to_string(),
            size_category: input.size_category.clone(),
            file: input.file.clone(),
            bytes: input.bytes,
            parser: parser_name.to_string(),
            input_length: input.token_count,
            token_count: input.token_count,
            median_time_ns: 0.0,
            mad_ns: 0.0,
            peak_memory_bytes: 0,
            iterations: 0,
            recognized: false,
            parse_correct: false,
            status: "PARSE_FAIL".to_string(),
            error_nodes: 0,
            missing_nodes: 0,
        }
    }

    fn timeout(input: &InputCase, parser_name: &str) -> Self {
        BenchmarkResult {
            language: "java_tree".to_string(),
            size_category: input.size_category.clone(),
            file: input.file.clone(),
            bytes: input.bytes,
            parser: parser_name.to_string(),
            input_length: input.token_count,
            token_count: input.token_count,
            median_time_ns: 0.0,
            mad_ns: 0.0,
            peak_memory_bytes: 0,
            iterations: 0,
            recognized: false,
            parse_correct: false,
            status: "TIMEOUT".to_string(),
            error_nodes: 0,
            missing_nodes: 0,
        }
    }

    fn to_worker_row(&self) -> String {
        format!(
            "{},{},{}",
            self.to_csv_row(),
            self.error_nodes,
            self.missing_nodes
        )
    }

    fn from_worker_row(row: &str) -> Result<Self, String> {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() != 16 {
            return Err(format!(
                "expected 16 worker result fields, found {}",
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
            error_nodes: fields[14]
                .parse()
                .map_err(|e| format!("invalid error-node count: {e}"))?,
            missing_nodes: fields[15]
                .parse()
                .map_err(|e| format!("invalid missing-node count: {e}"))?,
        })
    }
}

fn worker_result_from_output(
    exit_code: Option<i32>,
    stdout: &[u8],
    input: &InputCase,
    parser_name: &str,
) -> std::io::Result<BenchmarkResult> {
    if exit_code == Some(TIMEOUT_EXIT_CODE) {
        return Ok(BenchmarkResult::timeout(input, parser_name));
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
    BenchmarkResult::from_worker_row(payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn measure_peak_memory_with_result<T, F>(mut parse_fn: F) -> (T, usize)
where
    F: FnMut() -> T,
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
    F: FnMut() -> bool,
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

fn tree_to_bool(tree: Option<ParseTree>) -> bool {
    tree.is_some()
}

fn parse_without_errors(parser: &mut Parser, source: &str) -> bool {
    parser
        .parse(source, None)
        .map(|tree| !tree.root_node().has_error())
        .unwrap_or(false)
}

fn count_error_nodes(node: Node) -> (usize, usize) {
    let mut errors = usize::from(node.is_error());
    let mut missing = usize::from(node.is_missing());
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        let (child_errors, child_missing) = count_error_nodes(child);
        errors += child_errors;
        missing += child_missing;
    }

    (errors, missing)
}

fn collect_files(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(Path::new(dir))
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("java") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

fn load_inputs(files: &[PathBuf]) -> Vec<InputCase> {
    let lexer = Lexer::from_file(JAVA_TREE_LEXER_SPEC).unwrap_or_else(|e| {
        panic!(
            "Cannot load Java tree lexer spec {}: {}",
            JAVA_TREE_LEXER_SPEC, e
        )
    });
    let grammar = grammars::load_grammar_from_file(JAVA_TREE_GRAMMAR)
        .unwrap_or_else(|e| panic!("Cannot load Java tree grammar {}: {}", JAVA_TREE_GRAMMAR, e));

    let mut inputs = Vec::new();
    for path in files {
        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let source = match fs::read_to_string(path) {
            Ok(source) => source,
            Err(e) => {
                eprintln!("[WARN] Cannot read {:?}: {}", path, e);
                continue;
            }
        };
        let bytes = source.len();
        let ids = match lexer
            .lex(&source)
            .and_then(|tokens| encode(&tokens, &grammar))
        {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!(
                    "[WARN] Cannot lex/encode local java_tree tokens for {}: {}",
                    file, e
                );
                continue;
            }
        };
        let glr_ids = ids.iter().map(|&id| (id + 1) as i32).collect();
        let token_count = ids.len();

        inputs.push(InputCase {
            size_category: size_prefix(&file).to_string(),
            file,
            source,
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

fn benchmark_tree_sitter_input(parser: &mut Parser, input: &InputCase) -> BenchmarkResult {
    let ((tree, peak_memory_bytes), _) = run_with_hard_timeout(PARSE_TIMEOUT, || {
        measure_peak_memory_with_result(|| parser.parse(&input.source, None))
    });
    let (recognized, status, error_nodes, missing_nodes) = match tree {
        Some(tree) => {
            let (error_nodes, missing_nodes) = count_error_nodes(tree.root_node());
            let recognized = !tree.root_node().has_error();
            (
                recognized,
                if recognized { "OK" } else { "ERROR_TREE" },
                error_nodes,
                missing_nodes,
            )
        }
        None => (false, "NO_TREE", 0, 0),
    };
    let (median_time_ns, mad_ns, iterations) =
        measure(|| parse_without_errors(parser, &input.source));

    BenchmarkResult {
        language: "java_tree".to_string(),
        size_category: input.size_category.clone(),
        file: input.file.clone(),
        bytes: input.bytes,
        parser: "TreeSitterJava".to_string(),
        input_length: input.token_count,
        token_count: input.token_count,
        median_time_ns,
        mad_ns,
        peak_memory_bytes,
        iterations,
        recognized,
        parse_correct: recognized,
        status: status.to_string(),
        error_nodes,
        missing_nodes,
    }
}

fn benchmark_local_parser<F>(
    input: &InputCase,
    parser_name: &str,
    mut parse_fn: F,
) -> BenchmarkResult
where
    F: FnMut() -> bool,
{
    let ((recognized, peak_memory_bytes), _) = run_with_hard_timeout(PARSE_TIMEOUT, || {
        measure_peak_memory_with_result(&mut parse_fn)
    });
    let (median_time_ns, mad_ns, iterations) = measure(&mut parse_fn);
    if !recognized {
        let mut result = BenchmarkResult::parse_fail(input, parser_name);
        result.median_time_ns = median_time_ns;
        result.mad_ns = mad_ns;
        result.peak_memory_bytes = peak_memory_bytes;
        result.iterations = iterations;
        return result;
    }

    BenchmarkResult {
        language: "java_tree".to_string(),
        size_category: input.size_category.clone(),
        file: input.file.clone(),
        bytes: input.bytes,
        parser: parser_name.to_string(),
        input_length: input.token_count,
        token_count: input.token_count,
        median_time_ns,
        mad_ns,
        peak_memory_bytes,
        iterations,
        recognized: true,
        parse_correct: true,
        status: "OK".to_string(),
        error_nodes: 0,
        missing_nodes: 0,
    }
}

fn run_pair_worker(parser_name: &str, file: &str) -> Result<BenchmarkResult, String> {
    let path = Path::new(INPUT_DIR).join(file);
    let mut inputs = load_inputs(&[path]);
    let input = inputs
        .pop()
        .ok_or_else(|| format!("failed to load worker input {file}"))?;

    if parser_name == "TreeSitterJava" {
        let mut parser = Parser::new();
        let language: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
        parser
            .set_language(&language)
            .map_err(|e| format!("failed to load tree-sitter Java grammar: {e}"))?;
        return Ok(benchmark_tree_sitter_input(&mut parser, &input));
    }

    let grammar = grammars::load_grammar_from_file(JAVA_TREE_GRAMMAR)
        .map_err(|e| format!("failed to load Java tree grammar: {e}"))?;
    match parser_name {
        "Leo" => {
            let mut parser = earley_leo::LeoParser::new(grammar.clone());
            Ok(benchmark_local_parser(&input, parser_name, || {
                tree_to_bool(parser.parse(input.ids.clone()))
            }))
        }
        "GLL" => {
            let mut parser = gll::GLLParser::new(&grammar);
            Ok(benchmark_local_parser(&input, parser_name, || {
                parser.parse_one(&input.ids).is_some()
            }))
        }
        "RNGLR" => {
            let mut parser = glr::RnglrParser::import_table_from_csv(JAVA_TREE_GLR_TABLE)
                .map_err(|e| format!("failed to load RNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            Ok(benchmark_local_parser(&input, parser_name, || {
                tree_to_bool(parser.parse(&input.glr_ids))
            }))
        }
        "BRNGLR" => {
            let mut parser = glr::BrnglrParser::import_table_from_csv(JAVA_TREE_GLR_TABLE)
                .map_err(|e| format!("failed to load BRNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            Ok(benchmark_local_parser(&input, parser_name, || {
                tree_to_bool(parser.parse(&input.glr_ids))
            }))
        }
        _ => Err(format!("unknown parser: {parser_name}")),
    }
}

fn pair_worker_main(args: &[String]) -> i32 {
    if args.len() != 4 {
        eprintln!("Usage: {} {PAIR_WORKER_ARG} <parser> <file>", args[0]);
        return 2;
    }
    match run_pair_worker(&args[2], &args[3]) {
        Ok(result) => {
            println!("{WORKER_RESULT_PREFIX}{}", result.to_worker_row());
            0
        }
        Err(error) => {
            eprintln!("Benchmark worker failed: {error}");
            1
        }
    }
}

fn run_pair_in_worker(input: &InputCase, parser_name: &str) -> std::io::Result<BenchmarkResult> {
    let output = Command::new(std::env::current_exe()?)
        .args([PAIR_WORKER_ARG, parser_name, &input.file])
        .output()?;
    worker_result_from_output(output.status.code(), &output.stdout, input, parser_name).map_err(
        |error| {
            let stderr = String::from_utf8_lossy(&output.stderr);
            std::io::Error::new(error.kind(), format!("{error}; worker stderr: {stderr}"))
        },
    )
}

fn generate_glr_table(grammar: &grammars::NumericGrammar) -> std::io::Result<()> {
    println!("Generating GLR table: {}", JAVA_TREE_GLR_TABLE);
    fs::create_dir_all("table")?;
    let start = Instant::now();

    let done = Arc::new(AtomicBool::new(false));
    let done_for_thread = done.clone();
    let progress = thread::spawn(move || {
        while !done_for_thread.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(30));
            if !done_for_thread.load(Ordering::Relaxed) {
                println!(
                    "  ... still generating GLR table ({:.1?} elapsed)",
                    start.elapsed()
                );
            }
        }
    });

    let table_gen = table_generator::TableGenerator::new(grammar);
    let result = table_gen.export_to_csv_numeric(JAVA_TREE_GLR_TABLE);
    done.store(true, Ordering::Relaxed);
    let _ = progress.join();
    result?;

    println!("Generated GLR table in {:.2?}.", start.elapsed());
    Ok(())
}

fn parse_args() -> Result<RunOptions, String> {
    let mut options = RunOptions {
        regenerate_table: false,
        generate_table_only: false,
    };

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--regenerate-table" => options.regenerate_table = true,
            "--generate-table-only" => {
                options.regenerate_table = true;
                options.generate_table_only = true;
            }
            "-h" | "--help" => {
                return Err(format!(
                    "Usage:\n  cargo run --release --bin benchmark_tree_sitter_java\n  cargo run --release --bin benchmark_tree_sitter_java -- --generate-table-only\n  cargo run --release --bin benchmark_tree_sitter_java -- --regenerate-table"
                ));
            }
            other => return Err(format!("Unknown argument: {}", other)),
        }
    }

    Ok(options)
}

fn run_main() -> std::io::Result<()> {
    let options =
        parse_args().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

    println!("Tree-sitter Java + Local Parser Benchmark");
    println!("=========================================");
    println!("Input   : {}", INPUT_DIR);
    println!("Grammar : {}", JAVA_TREE_GRAMMAR);
    println!("Lexer   : {}", JAVA_TREE_LEXER_SPEC);
    println!("GLR CSV : {}", JAVA_TREE_GLR_TABLE);
    println!("Output  : {}", OUT_PATH);
    println!("Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("Timed iterations: {}", TIMED_ITERATIONS);
    println!("Per-invocation timeout: {:?}", PARSE_TIMEOUT);

    let grammar = grammars::load_grammar_from_file(JAVA_TREE_GRAMMAR)
        .unwrap_or_else(|e| panic!("Cannot load Java tree grammar {}: {}", JAVA_TREE_GRAMMAR, e));
    if options.regenerate_table || !Path::new(JAVA_TREE_GLR_TABLE).exists() {
        generate_glr_table(&grammar)?;
    } else {
        println!("Using existing GLR table: {}", JAVA_TREE_GLR_TABLE);
    }

    if options.generate_table_only {
        println!("Table generation complete; exiting before benchmark.");
        return Ok(());
    }

    let files = collect_files(INPUT_DIR);
    if files.is_empty() {
        eprintln!("[SKIP] No .java files found in {}.", INPUT_DIR);
        return Ok(());
    }

    let inputs = load_inputs(&files);
    println!("Found {} Java source files.", inputs.len());

    if inputs.is_empty() {
        eprintln!("[SKIP] No benchmarkable Java inputs found.");
        return Ok(());
    }

    fs::create_dir_all(RESULT_DIR)?;
    let mut csv_file = File::create(OUT_PATH)?;
    writeln!(
        csv_file,
        "language,size_category,file,bytes,\
         parser,input_length,token_count,median_time_ns,mad_ns,\
         peak_memory_bytes,iterations,recognized,parse_correct,status"
    )?;

    for (idx, input) in inputs.iter().enumerate() {
        let display_file = &input.file[..input.file.len().min(60)];
        println!(
            "\n  Input #{} [{}] {} bytes, {} local tokens",
            idx + 1,
            display_file,
            input.bytes,
            input.token_count
        );

        let result = run_pair_in_worker(input, "TreeSitterJava")?;
        if result.status == "TIMEOUT" {
            println!(
                "    [TIMEOUT] {:14}: invocation exceeded {:?}",
                result.parser, PARSE_TIMEOUT
            );
        } else {
            let recog_sym = if result.recognized { "R" } else { "!" };
            println!(
                "    [{}] {:14}: {:>12.0} ns +/- {:>8.0} ns  ({} iters, errors={}, missing={})",
                recog_sym,
                result.parser,
                result.median_time_ns,
                result.mad_ns,
                result.iterations,
                result.error_nodes,
                result.missing_nodes
            );
        }
        writeln!(csv_file, "{}", result.to_csv_row())?;
        csv_file.flush()?;

        for &parser_name in LOCAL_PARSERS {
            let result = run_pair_in_worker(input, parser_name)?;

            if result.status == "TIMEOUT" {
                println!(
                    "    [TIMEOUT] {:14}: invocation exceeded {:?}",
                    result.parser, PARSE_TIMEOUT
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            let recog_sym = if result.recognized { "R" } else { "!" };
            println!(
                "    [{}] {:14}: {:>12.0} ns +/- {:>8.0} ns  ({} iters, status={})",
                recog_sym,
                result.parser,
                result.median_time_ns,
                result.mad_ns,
                result.iterations,
                result.status
            );
            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?;
        }
    }

    println!("\nWritten to {}", OUT_PATH);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some(PAIR_WORKER_ARG) {
        let exit_code = thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(move || pair_worker_main(&args))
            .expect("Failed to spawn worker thread with larger stack")
            .join()
            .expect("Worker thread panicked");
        std::process::exit(exit_code);
    }

    let result = thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(run_main)
        .expect("Failed to spawn benchmark thread")
        .join()
        .expect("Benchmark thread panicked");

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT_CHILD_ENV: &str = "BENCHMARK_TREE_SITTER_JAVA_TIMEOUT_CHILD";

    #[test]
    fn output_path_uses_benchmark_tree_sitter_java_result_folder() {
        assert_eq!(
            OUT_PATH,
            "results/benchmark_tree_sitter_java/benchmark_tree_sitter_java.csv"
        );
    }

    #[test]
    fn measure_runs_exactly_ten_timed_iterations() {
        let mut calls = 0;
        let (_, _, iterations) = measure(|| {
            calls += 1;
            true
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
            source: String::new(),
            bytes: 42,
            ids: vec![],
            glr_ids: vec![],
            token_count: 17,
        };
        let result = worker_result_from_output(Some(124), b"", &input, "TreeSitterJava").unwrap();
        assert_eq!(result.parser, "TreeSitterJava");
        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, "TIMEOUT");
    }

    #[test]
    fn parse_failure_still_runs_ten_timed_iterations() {
        let input = InputCase {
            file: "A.java".to_string(),
            size_category: "S".to_string(),
            source: String::new(),
            bytes: 0,
            ids: vec![],
            glr_ids: vec![],
            token_count: 0,
        };
        let mut calls = 0;
        let result = benchmark_local_parser(&input, "Test", || {
            calls += 1;
            false
        });
        assert_eq!(calls, 11);
        assert_eq!(result.iterations, 10);
        assert_eq!(result.status, "PARSE_FAIL");
    }
}
