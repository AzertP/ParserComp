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
//!   results/benchmark_tree_sitter_java.csv

use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{earley_leo, gll, glr};
use std::collections::HashSet;
use std::env;
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
use tree_sitter::{Node, Parser};

const INPUT_DIR: &str = "input/code/Java";
const JAVA_TREE_LEXER_SPEC: &str = "grammars/lexer/java_tree_regex.json";
const JAVA_TREE_GRAMMAR: &str = "grammars/java_tree.json";
const JAVA_TREE_GLR_TABLE: &str = "table/java_tree_glr_table.csv";
const OUT_PATH: &str = "results/benchmark_tree_sitter_java.csv";
const LOCAL_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];

const WARMUP_ITERATIONS: u32 = 1;
const MIN_ITERATIONS: u32 = 10;
const MAX_ITERATIONS: u32 = 20;
const TARGET_TIME: Duration = Duration::from_millis(500);
const TIMEOUT_THRESHOLD: f64 = 1_000_000_000.0;

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
}

fn measure_peak_memory<F>(mut parse_fn: F) -> usize
where
    F: FnMut() -> bool,
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
    F: FnMut() -> bool,
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
    let tree = parser
        .parse(&input.source, None)
        .expect("tree-sitter parser returned no tree");
    let (error_nodes, missing_nodes) = count_error_nodes(tree.root_node());
    let recognized = !tree.root_node().has_error();

    let peak_memory_bytes = measure_peak_memory(|| parse_without_errors(parser, &input.source));
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
        status: if recognized { "OK" } else { "ERROR_TREE" }.to_string(),
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
    if !parse_fn() {
        return BenchmarkResult::parse_fail(input, parser_name);
    }

    let peak_memory_bytes = measure_peak_memory(&mut parse_fn);
    let (median_time_ns, mad_ns, iterations) = measure(&mut parse_fn);

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

    let mut tree_sitter_parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
    tree_sitter_parser
        .set_language(&language)
        .expect("Error loading tree-sitter Java grammar");

    let mut leo = earley_leo::LeoParser::new(grammar.clone());
    let mut gll_parser = gll::GLLParser::new(&grammar);
    let mut rnglr = glr::RnglrParser::import_table_from_csv(JAVA_TREE_GLR_TABLE)
        .expect("Failed to load RNGLR table");
    let mut brnglr = glr::BrnglrParser::import_table_from_csv(JAVA_TREE_GLR_TABLE)
        .expect("Failed to load BRNGLR table");
    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    fs::create_dir_all("results")?;
    let mut csv_file = File::create(OUT_PATH)?;
    writeln!(
        csv_file,
        "language,size_category,file,bytes,\
         parser,input_length,token_count,median_time_ns,mad_ns,\
         peak_memory_bytes,iterations,recognized,parse_correct,status"
    )?;

    let mut timed_out: HashSet<String> = HashSet::new();

    for (idx, input) in inputs.iter().enumerate() {
        println!(
            "\n  Input #{} [{:>3}] {} bytes, {} local tokens",
            idx + 1,
            input.size_category,
            input.bytes,
            input.token_count
        );
        println!("  {}", &input.file[..input.file.len().min(60)]);

        let result = benchmark_tree_sitter_input(&mut tree_sitter_parser, input);
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
        writeln!(csv_file, "{}", result.to_csv_row())?;
        csv_file.flush()?;

        for &parser_name in LOCAL_PARSERS {
            if timed_out.contains(parser_name) {
                let result = BenchmarkResult::timeout(input, parser_name);
                println!(
                    "    [T] {:14}: previously timed out, skipping",
                    result.parser
                );
                writeln!(csv_file, "{}", result.to_csv_row())?;
                csv_file.flush()?;
                continue;
            }

            let result = match parser_name {
                "Leo" => benchmark_local_parser(input, "Leo", || {
                    tree_to_bool(leo.parse(input.ids.clone()))
                }),
                "GLL" => benchmark_local_parser(input, "GLL", || {
                    // Benchmark GLL recognition here. `parse()` and `parse_all()`
                    // remain available for callers that need tree extraction.
                    gll_parser.parse_on(input.ids.clone())
                }),
                "RNGLR" => benchmark_local_parser(input, "RNGLR", || {
                    tree_to_bool(rnglr.parse(&input.glr_ids))
                }),
                "BRNGLR" => benchmark_local_parser(input, "BRNGLR", || {
                    tree_to_bool(brnglr.parse(&input.glr_ids))
                }),
                _ => continue,
            };

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

            if result.median_time_ns > TIMEOUT_THRESHOLD {
                println!(
                    "    [T] {:14}: exceeded timeout threshold, skipping larger inputs",
                    result.parser
                );
                timed_out.insert(result.parser.clone());
            }
        }
    }

    println!("\nWritten to {}", OUT_PATH);
    Ok(())
}

fn main() {
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
