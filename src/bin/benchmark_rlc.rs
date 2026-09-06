//! Parser benchmark for referenceLanguageCorpora inputs.
//!
//! For each configured RLC language, discovers files directly under
//! `input/rlc/<language>/`, lexes each file, encodes tokens against the
//! tokenized grammar, and benchmarks Leo, GLL, RNGLR, and BRNGLR.
//!
//! Usage:
//!   cargo run --release --bin benchmark_rlc
//!
//! Output:
//!   results/benchmark_rlc/benchmark_rlc_<language>.csv

use memory_stats::memory_stats;
use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{earley_leo, gll, glr};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// RLC configurations
// ============================================================================

const DEFAULT_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];
const RLC_MANIFEST_PATH: &str = "input/rlc/manifest.csv";

struct RlcConfig {
    name: &'static str,
    corpus: &'static str,
    lexer_spec: &'static str,
    grammar_path: &'static str,
    input_dir: &'static str,
    table_path: &'static str,
    generate_table: bool,
    parsers: &'static [&'static str],
}

const CONFIGS: &[RlcConfig] = &[
    // RlcConfig {
    //     name: "sml",
    //     corpus: "MLWorks",
    //     lexer_spec: "grammars/lexer/sml_regex.json",
    //     grammar_path: "grammars/sml_tok.json",
    //     input_dir: "input/rlc/sml",
    //     table_path: "table/sml_tok_glr_table.csv",
    //     generate_table: false,
    //     parsers: DEFAULT_PARSERS,
    // },
    RlcConfig {
        name: "java",
        corpus: "org",
        lexer_spec: "grammars/lexer/java_regex.json",
        grammar_path: "grammars/java_tok.json",
        input_dir: "input/rlc/java",
        table_path: "table/java_tok_glr_table.csv",
        generate_table: false,
        parsers: DEFAULT_PARSERS,
    },
    // RlcConfig {
    //     name: "gamma2",
    //     corpus: "org",
    //     lexer_spec: "grammars/lexer/gamma2_regex.json",
    //     grammar_path: "grammars/gamma2_tok.json",
    //     input_dir: "input/rlc/gamma2",
    //     table_path: "table/gamma2_tok_glr_table.csv",
    //     generate_table: true,
    //     parsers: DEFAULT_PARSERS,
    // },
    // RlcConfig {
    //     name: "gamma3",
    //     corpus: "org",
    //     lexer_spec: "grammars/lexer/gamma3_regex.json",
    //     grammar_path: "grammars/gamma3_tok.json",
    //     input_dir: "input/rlc/gamma3",
    //     table_path: "table/gamma3_tok_glr_table.csv",
    //     generate_table: true,
    //     parsers: DEFAULT_PARSERS,
    // },
];

// ============================================================================
// Timing constants
// ============================================================================

const WARMUP_ITERATIONS: u32 = 1;
const TIMED_ITERATIONS: u32 = 10;
const PARSE_TIMEOUT: Duration = Duration::from_secs(1);
const TIMEOUT_EXIT_CODE: i32 = 124;
const PAIR_WORKER_ARG: &str = "--pair-worker";
const WORKER_RESULT_PREFIX: &str = "WORKER_RESULT\t";
const RESULT_DIR: &str = "results/benchmark_rlc";

fn output_path(config_name: &str) -> String {
    format!("{}/benchmark_rlc_{}.csv", RESULT_DIR, config_name)
}

// ============================================================================
// Expected-result manifest and conformance tracking
// ============================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpectedResult {
    Accept,
    Reject,
}

impl ExpectedResult {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "accept" | "accepted" => Some(ExpectedResult::Accept),
            "reject" | "rejected" => Some(ExpectedResult::Reject),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ExpectedResult::Accept => "accept",
            ExpectedResult::Reject => "reject",
        }
    }

    fn matches_recognition(self, recognized: bool) -> bool {
        matches!(
            (self, recognized),
            (ExpectedResult::Accept, true) | (ExpectedResult::Reject, false)
        )
    }
}

#[derive(Default)]
struct ExpectedResults {
    by_input_path: HashMap<String, ExpectedResult>,
    by_language_file: HashMap<(String, String), ExpectedResult>,
}

#[derive(Deserialize)]
struct ManifestRecord {
    language: String,
    file: String,
    input_path: String,
    expected_result: String,
}

impl ExpectedResults {
    fn from_reader<R: Read>(reader: R) -> csv::Result<Self> {
        let mut csv_reader = csv::Reader::from_reader(reader);
        let mut expected_results = ExpectedResults::default();

        for record in csv_reader.deserialize::<ManifestRecord>() {
            let record = record?;
            let Some(expected_result) = ExpectedResult::parse(&record.expected_result) else {
                continue;
            };

            expected_results
                .by_input_path
                .insert(normalize_manifest_path(&record.input_path), expected_result);
            expected_results
                .by_language_file
                .insert((record.language, record.file), expected_result);
        }

        Ok(expected_results)
    }

    fn from_path(path: &Path) -> csv::Result<Self> {
        Self::from_reader(File::open(path)?)
    }

    fn get(&self, input_path: &str, language: &str, file: &str) -> Option<ExpectedResult> {
        self.by_input_path
            .get(&normalize_manifest_path(input_path))
            .copied()
            .or_else(|| {
                self.by_language_file
                    .get(&(language.to_string(), file.to_string()))
                    .copied()
            })
    }

    fn is_empty(&self) -> bool {
        self.by_input_path.is_empty() && self.by_language_file.is_empty()
    }

    fn len(&self) -> usize {
        self.by_input_path.len()
    }
}

fn load_expected_results(path: &str) -> ExpectedResults {
    match ExpectedResults::from_path(Path::new(path)) {
        Ok(expected_results) => {
            println!(
                "Loaded {} expected RLC results from {}.",
                expected_results.len(),
                path
            );
            expected_results
        }
        Err(e) => {
            eprintln!(
                "[WARN] Could not load expected-result manifest {}: {}",
                path, e
            );
            ExpectedResults::default()
        }
    }
}

fn normalize_manifest_path(path: &str) -> String {
    path.trim_start_matches("./").replace('\\', "/")
}

fn conformance_for(
    expected_result: Option<ExpectedResult>,
    recognized: bool,
    status: &str,
) -> Option<bool> {
    if status == "TIMEOUT" {
        None
    } else {
        expected_result.map(|expected| expected.matches_recognition(recognized))
    }
}

fn expected_result_csv(expected_result: Option<ExpectedResult>) -> &'static str {
    expected_result.map(ExpectedResult::as_str).unwrap_or("")
}

fn conformance_csv(conforms: Option<bool>) -> &'static str {
    match conforms {
        Some(true) => "true",
        Some(false) => "false",
        None => "",
    }
}

#[derive(Clone, Default)]
struct ConformanceStats {
    known: usize,
    conforming: usize,
    unknown: usize,
}

impl ConformanceStats {
    fn record(&mut self, conforms: Option<bool>) {
        match conforms {
            Some(true) => {
                self.known += 1;
                self.conforming += 1;
            }
            Some(false) => {
                self.known += 1;
            }
            None => {
                self.unknown += 1;
            }
        }
    }

    fn merge(&mut self, other: &ConformanceStats) {
        self.known += other.known;
        self.conforming += other.conforming;
        self.unknown += other.unknown;
    }

    fn nonconforming(&self) -> usize {
        self.known.saturating_sub(self.conforming)
    }

    fn rate(&self) -> f64 {
        if self.known == 0 {
            0.0
        } else {
            (self.conforming as f64 / self.known as f64) * 100.0
        }
    }
}

#[derive(Clone, Default)]
struct ConformanceSummary {
    overall: ConformanceStats,
    by_parser: BTreeMap<String, ConformanceStats>,
}

impl ConformanceSummary {
    fn record(&mut self, result: &BenchmarkResult) {
        self.overall.record(result.conforms);
        self.by_parser
            .entry(result.parser.clone())
            .or_default()
            .record(result.conforms);
    }

    fn merge(&mut self, other: &ConformanceSummary) {
        self.overall.merge(&other.overall);
        for (parser, stats) in &other.by_parser {
            self.by_parser
                .entry(parser.clone())
                .or_default()
                .merge(stats);
        }
    }

    fn has_rows(&self) -> bool {
        self.overall.known > 0 || self.overall.unknown > 0
    }
}

fn print_conformance_summary(label: &str, summary: &ConformanceSummary) {
    if !summary.has_rows() {
        return;
    }

    println!("\nConformance summary for {}:", label);
    print_conformance_stats("Overall", &summary.overall);
    for (parser, stats) in &summary.by_parser {
        print_conformance_stats(parser, stats);
    }
}

fn print_conformance_stats(label: &str, stats: &ConformanceStats) {
    println!(
        "  {:8}: {}/{} conforming ({:.2}%), {} nonconforming, {} unknown",
        label,
        stats.conforming,
        stats.known,
        stats.rate(),
        stats.nonconforming(),
        stats.unknown
    );
}

// ============================================================================
// Benchmark result
// ============================================================================

#[derive(Clone)]
struct BenchmarkResult {
    language: String,
    corpus: String,
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
    expected_result: Option<ExpectedResult>,
    conforms: Option<bool>,
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.2},{:.2},{},{},{},{},{},{},{}",
            self.language,
            self.corpus,
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
            expected_result_csv(self.expected_result),
            conformance_csv(self.conforms),
        )
    }

    fn timeout(
        lang: &str,
        corpus: &str,
        file: &str,
        bytes: usize,
        parser_name: &str,
        token_count: usize,
        expected_result: Option<ExpectedResult>,
    ) -> Self {
        let status = "TIMEOUT";
        BenchmarkResult {
            language: lang.to_string(),
            corpus: corpus.to_string(),
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
            status: status.to_string(),
            expected_result,
            conforms: conformance_for(expected_result, false, status),
        }
    }

    fn parse_fail(
        lang: &str,
        corpus: &str,
        file: &str,
        bytes: usize,
        parser_name: &str,
        token_count: usize,
        expected_result: Option<ExpectedResult>,
    ) -> Self {
        let status = "PARSE_FAIL";
        BenchmarkResult {
            language: lang.to_string(),
            corpus: corpus.to_string(),
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
            status: status.to_string(),
            expected_result,
            conforms: conformance_for(expected_result, false, status),
        }
    }

    fn from_csv_row(row: &str) -> Result<Self, String> {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() != 16 {
            return Err(format!(
                "expected 16 worker result fields, found {}",
                fields.len()
            ));
        }
        let expected_result = ExpectedResult::parse(fields[14]);
        let conforms = match fields[15] {
            "true" => Some(true),
            "false" => Some(false),
            "" => None,
            other => return Err(format!("invalid conformance flag: {other}")),
        };
        Ok(BenchmarkResult {
            language: fields[0].to_string(),
            corpus: fields[1].to_string(),
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
            expected_result,
            conforms,
        })
    }
}

fn worker_result_from_output(
    exit_code: Option<i32>,
    stdout: &[u8],
    cfg: &RlcConfig,
    input: &InputCase,
    parser_name: &str,
) -> std::io::Result<BenchmarkResult> {
    if exit_code == Some(TIMEOUT_EXIT_CODE) {
        return Ok(BenchmarkResult::timeout(
            cfg.name,
            cfg.corpus,
            &input.file,
            input.bytes,
            parser_name,
            input.token_count,
            input.expected_result,
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
// Measurement functions
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
    corpus: &str,
    file: &str,
    bytes: usize,
    parser_name: &str,
    token_count: usize,
    expected_result: Option<ExpectedResult>,
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
        let mut result = BenchmarkResult::parse_fail(
            lang,
            corpus,
            file,
            bytes,
            parser_name,
            token_count,
            expected_result,
        );
        result.median_time_ns = median;
        result.mad_ns = mad;
        result.peak_memory_bytes = peak_mem;
        result.iterations = iters;
        return result;
    }

    let status = "OK";
    BenchmarkResult {
        language: lang.to_string(),
        corpus: corpus.to_string(),
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
        status: status.to_string(),
        expected_result,
        conforms: conformance_for(expected_result, true, status),
    }
}

// ============================================================================
// Input loading
// ============================================================================

struct InputCase {
    file: String,
    input_path: String,
    bytes: usize,
    ids: Vec<u32>,
    glr_ids: Vec<i32>,
    token_count: usize,
    expected_result: Option<ExpectedResult>,
}

fn collect_files(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(Path::new(dir))
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.is_file() {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

fn load_inputs(
    files: &[PathBuf],
    lexer: &Lexer,
    grammar: &grammars::NumericGrammar,
    expected_results: &ExpectedResults,
    language: &str,
) -> Vec<InputCase> {
    let mut inputs = Vec::new();

    for path in files {
        let input_path = normalize_manifest_path(&path.to_string_lossy());
        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let expected_result = expected_results.get(&input_path, language, &file);
        if expected_result.is_none() && !expected_results.is_empty() {
            eprintln!("[WARN] No expected result in manifest for {}.", input_path);
        }

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
            input_path,
            bytes,
            ids,
            glr_ids,
            token_count,
            expected_result,
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

fn run_pair_worker(
    cfg: &RlcConfig,
    parser_name: &str,
    file: &str,
) -> Result<BenchmarkResult, String> {
    let lexer =
        Lexer::from_file(cfg.lexer_spec).map_err(|e| format!("failed to load lexer: {e}"))?;
    let grammar = grammars::load_grammar_from_file(cfg.grammar_path)
        .map_err(|e| format!("failed to load grammar: {e}"))?;
    let expected_results = load_expected_results(RLC_MANIFEST_PATH);
    let path = Path::new(cfg.input_dir).join(file);
    let mut inputs = load_inputs(&[path], &lexer, &grammar, &expected_results, cfg.name);
    let input = inputs
        .pop()
        .ok_or_else(|| format!("failed to load worker input {file}"))?;

    match parser_name {
        "Leo" => {
            let mut parser = earley_leo::LeoParser::new(grammar.clone());
            Ok(benchmark_parser(
                cfg.name,
                cfg.corpus,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                input.expected_result,
                || parser.parse(input.ids.clone()),
            ))
        }
        "GLL" => {
            let mut parser = gll::GLLParser::new(&grammar);
            Ok(benchmark_parser(
                cfg.name,
                cfg.corpus,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                input.expected_result,
                || parser.parse_one(&input.ids),
            ))
        }
        "RNGLR" => {
            let mut parser = glr::RnglrParser::import_table_from_csv(cfg.table_path)
                .map_err(|e| format!("failed to load RNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            Ok(benchmark_parser(
                cfg.name,
                cfg.corpus,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                input.expected_result,
                || parser.parse(&input.glr_ids),
            ))
        }
        "BRNGLR" => {
            let mut parser = glr::BrnglrParser::import_table_from_csv(cfg.table_path)
                .map_err(|e| format!("failed to load BRNGLR table: {e}"))?;
            parser.set_grammar(grammar.clone());
            Ok(benchmark_parser(
                cfg.name,
                cfg.corpus,
                &input.file,
                input.bytes,
                parser_name,
                input.token_count,
                input.expected_result,
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
    cfg: &RlcConfig,
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
// Per-language benchmark
// ============================================================================

fn run_config(
    cfg: &RlcConfig,
    expected_results: &ExpectedResults,
) -> std::io::Result<ConformanceSummary> {
    println!("\n{}", "=".repeat(60));
    println!("Benchmarking RLC: {} ({})", cfg.name, cfg.corpus);
    println!("  Grammar : {}", cfg.grammar_path);
    println!("  Lexer   : {}", cfg.lexer_spec);
    println!("  Input   : {}", cfg.input_dir);
    println!("  Table   : {}", cfg.table_path);
    println!("{}", "=".repeat(60));

    let lexer = Lexer::from_file(cfg.lexer_spec)
        .unwrap_or_else(|e| panic!("Cannot load lexer spec {}: {}", cfg.lexer_spec, e));
    let grammar = grammars::load_grammar_from_file(cfg.grammar_path)
        .unwrap_or_else(|e| panic!("Cannot load grammar {}: {}", cfg.grammar_path, e));

    println!(
        "Grammar loaded ({} terminals, {} non-terminals).",
        grammar.num_terminals(),
        grammar.num_non_terminals()
    );

    if cfg.generate_table {
        println!("Generating GLR table...");
        fs::create_dir_all("table")?;
        let table_gen = table_generator::TableGenerator::new(&grammar);
        table_gen
            .export_to_csv_numeric(cfg.table_path)
            .expect("Failed to export GLR table");
        println!("GLR table generated.");
    } else {
        println!("Skipping table generation (using existing table)...");
    }

    let files = collect_files(cfg.input_dir);
    if files.is_empty() {
        eprintln!("[SKIP] No RLC files found in {}.", cfg.input_dir);
        return Ok(ConformanceSummary::default());
    }
    println!("{} RLC files found.", files.len());

    let inputs = load_inputs(&files, &lexer, &grammar, expected_results, cfg.name);
    if inputs.is_empty() {
        eprintln!("[SKIP] No benchmarkable inputs found for {}.", cfg.name);
        return Ok(ConformanceSummary::default());
    }
    println!(
        "{} benchmarkable inputs ({} to {} tokens).",
        inputs.len(),
        inputs.first().map(|input| input.token_count).unwrap_or(0),
        inputs.last().map(|input| input.token_count).unwrap_or(0)
    );

    fs::create_dir_all(RESULT_DIR)?;
    let out_path = output_path(cfg.name);
    let mut csv_file = File::create(&out_path)?;
    writeln!(
        csv_file,
        "language,corpus,file,bytes,\
         parser,input_length,token_count,median_time_ns,mad_ns,\
         peak_memory_bytes,iterations,recognized,parse_correct,status,\
         expected_result,conforms"
    )?;
    println!("\nWriting results to: {}", out_path);

    let mut conformance_summary = ConformanceSummary::default();

    for (idx, input) in inputs.iter().enumerate() {
        println!(
            "\n  Input #{} [{}] {} bytes, {} tokens, expected {}",
            idx + 1,
            input.file,
            input.bytes,
            input.token_count,
            expected_result_csv(input.expected_result)
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
                conformance_summary.record(&result);
                continue;
            }

            let recog_sym = if result.recognized { "R" } else { "x" };
            let parse_sym = if result.parse_correct { "P" } else { "x" };
            let conform_sym = match result.conforms {
                Some(true) => "C",
                Some(false) => "!",
                None => "?",
            };
            println!(
                "    [{}|{}|{}] {:8}: {:>12.0} ns +/- {:>8.0} ns  ({} iters)",
                recog_sym,
                parse_sym,
                conform_sym,
                result.parser,
                result.median_time_ns,
                result.mad_ns,
                result.iterations
            );

            if result.conforms == Some(false) {
                eprintln!(
                    "    [NONCONF] {} {} {} but expected {}.",
                    result.parser,
                    if result.recognized {
                        "accepted"
                    } else {
                        "rejected"
                    },
                    input.input_path,
                    expected_result_csv(result.expected_result)
                );
            }

            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?;
            conformance_summary.record(&result);
        }
    }

    println!("\nWritten to {}", out_path);
    print_conformance_summary(cfg.name, &conformance_summary);
    Ok(conformance_summary)
}

// ============================================================================
// Main
// ============================================================================

fn run_main() {
    println!("RLC Lexer + Parser Benchmark Tool");
    println!("=================================");
    println!("Benchmarks Leo, GLL, RNGLR, and BRNGLR on input/rlc/<language>/ files\n");
    println!("Configuration:");
    println!("  Warmup iterations : {}", WARMUP_ITERATIONS);
    println!("  Timed iterations  : {}", TIMED_ITERATIONS);
    println!("  Per-invocation timeout: {:?}", PARSE_TIMEOUT);

    let expected_results = load_expected_results(RLC_MANIFEST_PATH);
    let mut overall_conformance = ConformanceSummary::default();

    for cfg in CONFIGS {
        match run_config(cfg, &expected_results) {
            Ok(summary) => overall_conformance.merge(&summary),
            Err(e) => eprintln!("[ERROR] {}: {}", cfg.name, e),
        }
    }

    print_conformance_summary("all RLC benchmarks", &overall_conformance);
    println!("\nBenchmarking complete.");
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
        .stack_size(128 * 1024 * 1024)
        .spawn(run_main)
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT_CHILD_ENV: &str = "BENCHMARK_RLC_TIMEOUT_CHILD";

    #[test]
    fn output_path_uses_benchmark_rlc_result_folder() {
        assert_eq!(
            output_path("java"),
            "results/benchmark_rlc/benchmark_rlc_java.csv"
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
            input_path: "input/rlc/java/A.java".to_string(),
            bytes: 42,
            ids: vec![],
            glr_ids: vec![],
            token_count: 17,
            expected_result: Some(ExpectedResult::Accept),
        };
        let result = worker_result_from_output(Some(124), b"", &CONFIGS[0], &input, "Leo").unwrap();
        assert_eq!(result.parser, "Leo");
        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, "TIMEOUT");
        assert_eq!(result.conforms, None);
    }

    #[test]
    fn parse_failure_still_runs_ten_timed_iterations() {
        let mut calls = 0;
        let result = benchmark_parser(
            "test",
            "corpus",
            "input",
            4,
            "Test",
            2,
            Some(ExpectedResult::Reject),
            || {
                calls += 1;
                None
            },
        );
        assert_eq!(calls, 11);
        assert_eq!(result.iterations, 10);
        assert_eq!(result.status, "PARSE_FAIL");
    }

    #[test]
    fn manifest_reader_loads_accept_and_reject_expectations() {
        let manifest = "\
language,corpus,variant,file,input_path,upstream_path,expected_result,evidence_kind,evidence_accept_count,evidence_reject_count,evidence_sources,notes
java,rhul,org,Ex2.java,input/rlc/java/Ex2.java,languages/java/corpus/rhul/org/Ex2.java,accept,,,,,
gamma2,rhul,org,b.001,input/rlc/gamma2/b.001,languages/gamma2/corpus/rhul/org/b.001,reject,,,,,
";

        let expectations = ExpectedResults::from_reader(manifest.as_bytes()).unwrap();

        assert_eq!(
            expectations.get("input/rlc/java/Ex2.java", "java", "Ex2.java"),
            Some(ExpectedResult::Accept)
        );
        assert_eq!(
            expectations.get("input/rlc/gamma2/b.001", "gamma2", "b.001"),
            Some(ExpectedResult::Reject)
        );
    }

    #[test]
    fn conformance_summary_counts_accepts_rejects_and_timeouts() {
        let mut summary = ConformanceSummary::default();
        let rows = [
            test_result("Leo", Some(ExpectedResult::Accept), true, "OK"),
            test_result("Leo", Some(ExpectedResult::Reject), false, "PARSE_FAIL"),
            test_result("GLL", Some(ExpectedResult::Accept), false, "PARSE_FAIL"),
            test_result("GLL", Some(ExpectedResult::Reject), false, "TIMEOUT"),
            test_result("RNGLR", None, true, "OK"),
        ];

        for result in &rows {
            summary.record(result);
        }

        assert_eq!(summary.overall.known, 3);
        assert_eq!(summary.overall.conforming, 2);
        assert_eq!(summary.overall.nonconforming(), 1);
        assert_eq!(summary.overall.unknown, 2);
        assert_eq!(summary.by_parser["Leo"].known, 2);
        assert_eq!(summary.by_parser["Leo"].conforming, 2);
        assert_eq!(summary.by_parser["GLL"].known, 1);
        assert_eq!(summary.by_parser["GLL"].conforming, 0);
        assert_eq!(summary.by_parser["GLL"].unknown, 1);
        assert_eq!(summary.by_parser["RNGLR"].unknown, 1);
    }

    #[test]
    fn rlc_configs_include_sml_corpus() {
        let sml = CONFIGS
            .iter()
            .find(|cfg| cfg.name == "sml")
            .expect("benchmark_rlc should include an SML config");

        assert_eq!(sml.corpus, "MLWorks");
        assert_eq!(sml.lexer_spec, "grammars/lexer/sml_regex.json");
        assert_eq!(sml.grammar_path, "grammars/sml_tok.json");
        assert_eq!(sml.input_dir, "input/rlc/sml");
        assert_eq!(sml.table_path, "table/sml_tok_glr_table.csv");
        assert!(sml.generate_table);
        assert_eq!(sml.parsers, DEFAULT_PARSERS);
    }

    fn test_result(
        parser: &str,
        expected_result: Option<ExpectedResult>,
        recognized: bool,
        status: &str,
    ) -> BenchmarkResult {
        BenchmarkResult {
            language: "java".to_string(),
            corpus: "org".to_string(),
            file: "Example.java".to_string(),
            bytes: 1,
            parser: parser.to_string(),
            input_length: 1,
            token_count: 1,
            median_time_ns: 0.0,
            mad_ns: 0.0,
            peak_memory_bytes: 0,
            iterations: 1,
            recognized,
            parse_correct: recognized,
            status: status.to_string(),
            expected_result,
            conforms: conformance_for(expected_result, recognized, status),
        }
    }
}
