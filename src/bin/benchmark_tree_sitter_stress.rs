//! Tree-sitter-only benchmark for the gamma2 and gamma3 stress grammars.
//!
//! Usage:
//!   cargo run --release --bin benchmark_tree_sitter_stress
//!   cargo run --release --bin benchmark_tree_sitter_stress -- --grammar gamma2
//!   cargo run --release --bin benchmark_tree_sitter_stress -- --grammar gamma3

use memory_stats::memory_stats;
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use tree_sitter::{Language, Node, Parser};

const RESULT_DIR: &str = "results/benchmark_tree_sitter_stress";
const CSV_HEADER: &str = "language,size_category,file,bytes,parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,recognized,parse_correct,status";

const WARMUP_ITERATIONS: u32 = 1;
const MIN_ITERATIONS: u32 = 10;
const MAX_ITERATIONS: u32 = 20;
const TARGET_TIME: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GrammarChoice {
    Gamma2,
    Gamma3,
}

#[derive(Debug)]
struct RunOptions {
    grammar: Option<GrammarChoice>,
}

#[derive(Clone, Copy)]
struct StressConfig {
    name: &'static str,
    input_dir: &'static str,
    output_path: &'static str,
    grammar: GrammarChoice,
}

const GAMMA2_CONFIG: StressConfig = StressConfig {
    name: "gamma2",
    input_dir: "input/rlc/gamma2",
    output_path: "results/benchmark_tree_sitter_stress/benchmark_tree_sitter_gamma2.csv",
    grammar: GrammarChoice::Gamma2,
};

const GAMMA3_CONFIG: StressConfig = StressConfig {
    name: "gamma3",
    input_dir: "input/rlc/gamma3",
    output_path: "results/benchmark_tree_sitter_stress/benchmark_tree_sitter_gamma3.csv",
    grammar: GrammarChoice::Gamma3,
};

#[derive(Clone)]
struct InputCase {
    file: String,
    size_category: String,
    source: String,
    bytes: usize,
    token_count: usize,
}

struct ParseInspection {
    recognized: bool,
    status: &'static str,
    error_nodes: usize,
    missing_nodes: usize,
}

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
}

fn usage() -> &'static str {
    "Usage:\n  cargo run --release --bin benchmark_tree_sitter_stress\n  cargo run --release --bin benchmark_tree_sitter_stress -- --grammar gamma2\n  cargo run --release --bin benchmark_tree_sitter_stress -- --grammar gamma3"
}

fn parse_grammar(value: &str) -> Result<GrammarChoice, String> {
    match value {
        "gamma2" => Ok(GrammarChoice::Gamma2),
        "gamma3" => Ok(GrammarChoice::Gamma3),
        other => Err(format!(
            "Unsupported stress grammar {other:?}; expected gamma2 or gamma3"
        )),
    }
}

fn parse_args_from<I, S>(args: I) -> Result<RunOptions, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    let _program = args.next();
    let mut grammar = None;

    while let Some(arg) = args.next() {
        let arg = arg.as_ref();
        match arg {
            "--grammar" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--grammar requires gamma2 or gamma3".to_string())?;
                grammar = Some(parse_grammar(value.as_ref())?);
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other if other.starts_with("--grammar=") => {
                grammar = Some(parse_grammar(&other["--grammar=".len()..])?);
            }
            other => return Err(format!("Unknown argument {other:?}\n{}", usage())),
        }
    }

    Ok(RunOptions { grammar })
}

fn selected_configs(options: &RunOptions) -> Vec<&'static StressConfig> {
    match options.grammar {
        None => vec![&GAMMA2_CONFIG, &GAMMA3_CONFIG],
        Some(GrammarChoice::Gamma2) => vec![&GAMMA2_CONFIG],
        Some(GrammarChoice::Gamma3) => vec![&GAMMA3_CONFIG],
    }
}

fn language_for(config: &StressConfig) -> Language {
    match config.grammar {
        GrammarChoice::Gamma2 => tree_sitter_gamma2::LANGUAGE.into(),
        GrammarChoice::Gamma3 => tree_sitter_gamma3::LANGUAGE.into(),
    }
}

fn parser_for(config: &StressConfig) -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(&language_for(config))
        .unwrap_or_else(|error| panic!("Error loading {} parser: {error}", config.name));
    parser
}

fn count_terminals(source: &str) -> usize {
    source
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .count()
}

fn collect_files(dir: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn size_category(file: &str) -> String {
    file.rsplit_once('.')
        .map(|(_, suffix)| suffix.to_string())
        .unwrap_or_else(|| "?".to_string())
}

fn load_inputs(files: &[PathBuf]) -> std::io::Result<Vec<InputCase>> {
    let mut inputs = Vec::with_capacity(files.len());
    for path in files {
        let source = fs::read_to_string(path)?;
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("?")
            .to_string();
        inputs.push(InputCase {
            size_category: size_category(&file),
            file,
            bytes: source.len(),
            token_count: count_terminals(&source),
            source,
        });
    }

    sort_inputs(&mut inputs);
    Ok(inputs)
}

fn sort_inputs(inputs: &mut [InputCase]) {
    inputs.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.file.cmp(&b.file)));
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

fn inspect_source(parser: &mut Parser, source: &str) -> ParseInspection {
    let Some(tree) = parser.parse(source, None) else {
        return ParseInspection {
            recognized: false,
            status: "NO_TREE",
            error_nodes: 0,
            missing_nodes: 0,
        };
    };
    let root = tree.root_node();
    let (error_nodes, missing_nodes) = count_error_nodes(root);
    let recognized = !root.has_error();
    ParseInspection {
        recognized,
        status: if recognized { "OK" } else { "ERROR_TREE" },
        error_nodes,
        missing_nodes,
    }
}

fn parse_without_errors(parser: &mut Parser, source: &str) -> bool {
    parser
        .parse(source, None)
        .map(|tree| !tree.root_node().has_error())
        .unwrap_or(false)
}

fn measure_peak_memory<F>(mut parse_fn: F) -> usize
where
    F: FnMut() -> bool,
{
    let start_mem = memory_stats().map(|usage| usage.physical_mem).unwrap_or(0);
    let peak_mem = Arc::new(AtomicUsize::new(start_mem));
    let stop_signal = Arc::new(AtomicBool::new(false));

    let thread_peak = peak_mem.clone();
    let thread_stop = stop_signal.clone();
    let sampler = thread::spawn(move || {
        while !thread_stop.load(Ordering::Relaxed) {
            if let Some(usage) = memory_stats() {
                thread_peak.fetch_max(usage.physical_mem, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    let _ = black_box(parse_fn());
    stop_signal.store(true, Ordering::Relaxed);
    let _ = sampler.join();

    peak_mem.load(Ordering::Relaxed).saturating_sub(start_mem)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

fn measure<F>(mut parse_fn: F) -> (f64, f64, u32)
where
    F: FnMut() -> bool,
{
    for _ in 0..WARMUP_ITERATIONS {
        let _ = black_box(parse_fn());
    }

    let measurement_start = Instant::now();
    let mut times = Vec::new();
    while times.len() < MAX_ITERATIONS as usize
        && (times.len() < MIN_ITERATIONS as usize || measurement_start.elapsed() < TARGET_TIME)
    {
        let start = Instant::now();
        let _ = black_box(parse_fn());
        times.push(start.elapsed().as_nanos() as f64);
    }

    let sample_count = times.len() as u32;
    let median_time = median(&mut times);
    let mut deviations: Vec<f64> = times
        .iter()
        .map(|time| (time - median_time).abs())
        .collect();
    let mad = median(&mut deviations);
    (median_time, mad, sample_count)
}

fn benchmark_input(
    parser: &mut Parser,
    config: &StressConfig,
    input: &InputCase,
) -> BenchmarkResult {
    let inspection = inspect_source(parser, &input.source);
    let peak_memory_bytes = measure_peak_memory(|| parse_without_errors(parser, &input.source));
    let (median_time_ns, mad_ns, iterations) =
        measure(|| parse_without_errors(parser, &input.source));

    BenchmarkResult {
        language: config.name.to_string(),
        size_category: input.size_category.clone(),
        file: input.file.clone(),
        bytes: input.bytes,
        parser: "TreeSitter".to_string(),
        input_length: input.token_count,
        token_count: input.token_count,
        median_time_ns,
        mad_ns,
        peak_memory_bytes,
        iterations,
        recognized: inspection.recognized,
        parse_correct: inspection.recognized,
        status: inspection.status.to_string(),
        error_nodes: inspection.error_nodes,
        missing_nodes: inspection.missing_nodes,
    }
}

fn ensure_expected_accepts(grammar: &str, failure_count: usize) -> std::io::Result<()> {
    if failure_count == 0 {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{grammar}: {failure_count} expected-accept inputs produced error trees or no tree"
            ),
        ))
    }
}

fn run_config(config: &StressConfig) -> std::io::Result<()> {
    let files = collect_files(config.input_dir)?;
    let inputs = load_inputs(&files)?;
    if inputs.is_empty() {
        eprintln!("[SKIP] No inputs found in {}", config.input_dir);
        return Ok(());
    }

    fs::create_dir_all(RESULT_DIR)?;
    let mut output = File::create(config.output_path)?;
    writeln!(output, "{CSV_HEADER}")?;

    println!(
        "\n{}: {} inputs from {}",
        config.name,
        inputs.len(),
        config.input_dir
    );
    let mut parser = parser_for(config);
    let mut recognition_failures = 0;
    for (index, input) in inputs.iter().enumerate() {
        let result = benchmark_input(&mut parser, config, input);
        if !result.recognized {
            recognition_failures += 1;
        }
        println!(
            "  {:>3}/{} {:<20} {:>12.0} ns ({} iterations, {}, errors={}, missing={})",
            index + 1,
            inputs.len(),
            input.file,
            result.median_time_ns,
            result.iterations,
            result.status,
            result.error_nodes,
            result.missing_nodes,
        );
        writeln!(output, "{}", result.to_csv_row())?;
        output.flush()?;
    }

    println!("Written to {}", config.output_path);
    ensure_expected_accepts(config.name, recognition_failures)
}

fn run_main() -> std::io::Result<()> {
    let options = parse_args_from(env::args())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    println!("Tree-sitter Stress Grammar Benchmark");
    println!("====================================");
    for config in selected_configs(&options) {
        run_config(config)?;
    }
    Ok(())
}

fn main() {
    let result = thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(run_main)
        .expect("Failed to spawn benchmark thread")
        .join()
        .expect("Benchmark thread panicked");

    if let Err(error) = result {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_grammar_argument_selects_both_grammars() {
        let options = parse_args_from(["benchmark"]).unwrap();
        assert_eq!(
            selected_configs(&options)
                .iter()
                .map(|config| config.name)
                .collect::<Vec<_>>(),
            vec!["gamma2", "gamma3"]
        );
    }

    #[test]
    fn grammar_argument_selects_one_grammar() {
        for (argument, expected) in [("gamma2", "gamma2"), ("gamma3", "gamma3")] {
            let options = parse_args_from(["benchmark", "--grammar", argument]).unwrap();
            let configs = selected_configs(&options);
            assert_eq!(configs.len(), 1);
            assert_eq!(configs[0].name, expected);
        }
    }

    #[test]
    fn unsupported_grammar_is_an_error() {
        let error = parse_args_from(["benchmark", "--grammar", "java"]).unwrap_err();
        assert!(error.contains("java"));
    }

    #[test]
    fn output_paths_are_separate_and_stable() {
        assert_eq!(
            GAMMA2_CONFIG.output_path,
            "results/benchmark_tree_sitter_stress/benchmark_tree_sitter_gamma2.csv"
        );
        assert_eq!(
            GAMMA3_CONFIG.output_path,
            "results/benchmark_tree_sitter_stress/benchmark_tree_sitter_gamma3.csv"
        );
    }

    #[test]
    fn terminal_count_ignores_ascii_whitespace() {
        assert_eq!(count_terminals(" b b b a\n\t"), 4);
    }

    #[test]
    fn inputs_sort_by_bytes_then_filename() {
        let mut inputs = vec![
            InputCase {
                file: "z.002".to_string(),
                size_category: "002".to_string(),
                source: "bb".to_string(),
                bytes: 2,
                token_count: 2,
            },
            InputCase {
                file: "b.001".to_string(),
                size_category: "001".to_string(),
                source: "b".to_string(),
                bytes: 1,
                token_count: 1,
            },
            InputCase {
                file: "a.002".to_string(),
                size_category: "002".to_string(),
                source: "bb".to_string(),
                bytes: 2,
                token_count: 2,
            },
        ];

        sort_inputs(&mut inputs);

        assert_eq!(
            inputs
                .iter()
                .map(|input| input.file.as_str())
                .collect::<Vec<_>>(),
            vec!["b.001", "a.002", "z.002"]
        );
    }

    #[test]
    fn valid_and_invalid_gamma2_inputs_are_classified() {
        let mut parser = parser_for(&GAMMA2_CONFIG);
        let valid = inspect_source(&mut parser, "bbba");
        assert!(valid.recognized);
        assert_eq!(valid.status, "OK");
        assert_eq!((valid.error_nodes, valid.missing_nodes), (0, 0));

        let invalid = inspect_source(&mut parser, "b");
        assert!(!invalid.recognized);
        assert_ne!(invalid.status, "OK");
    }

    #[test]
    fn valid_and_invalid_gamma3_inputs_are_classified() {
        let mut parser = parser_for(&GAMMA3_CONFIG);
        let valid = inspect_source(&mut parser, "bbb");
        assert!(valid.recognized);
        assert_eq!(valid.status, "OK");
        assert_eq!((valid.error_nodes, valid.missing_nodes), (0, 0));

        let invalid = inspect_source(&mut parser, "a");
        assert!(!invalid.recognized);
        assert_ne!(invalid.status, "OK");
    }

    #[test]
    fn csv_schema_and_parser_label_are_stable() {
        assert_eq!(
            CSV_HEADER,
            "language,size_category,file,bytes,parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,recognized,parse_correct,status"
        );

        let result = BenchmarkResult {
            language: "gamma2".to_string(),
            size_category: "006".to_string(),
            file: "bb_bbba.006".to_string(),
            bytes: 12,
            parser: "TreeSitter".to_string(),
            input_length: 6,
            token_count: 6,
            median_time_ns: 10.0,
            mad_ns: 1.0,
            peak_memory_bytes: 0,
            iterations: 10,
            recognized: true,
            parse_correct: true,
            status: "OK".to_string(),
            error_nodes: 0,
            missing_nodes: 0,
        };
        assert!(result
            .to_csv_row()
            .starts_with("gamma2,006,bb_bbba.006,12,TreeSitter,"));
    }

    #[test]
    fn expected_accept_failures_are_reported() {
        assert!(ensure_expected_accepts("gamma2", 0).is_ok());

        let error = ensure_expected_accepts("gamma2", 2).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("gamma2"));
        assert!(error.to_string().contains('2'));
    }
}
