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
use parser_comparison::parsers::gll::ll;
use parser_comparison::parsers::glr::lr;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{cyk, earley, gll, glr, valiant};
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
    input_path: &'static str,
    table_path: &'static str,
    lr_table_path: &'static str,
    generate_table: bool,
    parsers: &'static [&'static str],
}

const GENERAL_PARSERS: &[&str] = &["Earley", "GLL", "RNGLR", "BRNGLR", "Valiant", "CYK"];
const FAST_PARSERS: &[&str] = &["Earley", "GLL", "RNGLR", "BRNGLR"];
const GLL_GLR_PARSERS: &[&str] = &["GLL", "RNGLR", "BRNGLR"];
const GENERAL_PLUS_LR: &[&str] = &["Earley", "GLL", "RNGLR", "BRNGLR", "LR"];
const GENERAL_PLUS_LLLR: &[&str] = &["Earley", "GLL", "RNGLR", "BRNGLR", "LL", "LR"];
const NO_EARLEY_LLLR: &[&str] = &["GLL", "RNGLR", "BRNGLR", "LL", "LR"];

const CONFIGS: &[GrammarConfig] = &[
    GrammarConfig {
        name: "ambi_stress",
        grammar_path: "grammars/ambi.json",
        input_path: "input/ambi_stress.txt",
        table_path: "table/ambi_glr_table.csv",
        lr_table_path: "table/ambi_lr_table.csv",
        generate_table: true,
        parsers: GENERAL_PARSERS,
    },
    // GrammarConfig {
    //     name: "lr_json_large",
    //     grammar_path: "grammars/lr_json.json",
    //     input_path: "input/json_large.txt",
    //     table_path: "table/lr_json_glr_table.csv",
    //     lr_table_path: "table/lr_json_lr_table.csv",
    //     generate_table: true,
    //     parsers: GENERAL_PLUS_LR,
    // },
    // GrammarConfig {
    //     name: "lr_tinyc_large",
    //     grammar_path: "grammars/lr_tinyc.json",
    //     input_path: "input/tinyc_lr_large.txt",
    //     table_path: "table/lr_tinyc_glr_table.csv",
    //     lr_table_path: "table/lr_tinyc_lr_table.csv",
    //     generate_table: true,
    //     parsers: GENERAL_PLUS_LR,
    // },
    GrammarConfig {
        name: "json_with_lllr_large",
        grammar_path: "grammars/ll1_json.json",
        input_path: "input/json_ultra.txt",
        table_path: "table/json_ll1_table.csv",
        lr_table_path: "table/json_ll1_lr_table.csv",
        generate_table: true,
        parsers: NO_EARLEY_LLLR,
    },
    // GrammarConfig{
    //     name: "json_tokenized_with_lllr_large",
    //     grammar_path: "grammars/json_tokenized.json",
    //     input_path: "input/json_tokenized_large.txt",
    //     table_path: "table/json_tokenized_table.csv",
    //     lr_table_path: "table/json_tokenized_lr_table.csv",
    //     generate_table: true,
    // },
    GrammarConfig {
        name: "calc_with_lllr_large",
        grammar_path: "grammars/ll1_calc.json",
        input_path: "input/calc_large.txt",
        table_path: "table/calc_ll1_table.csv",
        lr_table_path: "table/calc_ll1_lr_table.csv",
        generate_table: true,
        parsers: GENERAL_PLUS_LLLR,
    },
    GrammarConfig {
        name: "sexp_with_lllr_large",
        grammar_path: "grammars/ll1_sexp.json",
        input_path: "input/sexp_ll1.txt",
        table_path: "table/sexp_ll1_table.csv",
        lr_table_path: "table/sexp_ll1_lr_table.csv",
        generate_table: true,
        parsers: GENERAL_PLUS_LLLR,
    },
    GrammarConfig {
        name: "tinypascal_with_lllr_large",
        grammar_path: "grammars/ll1_tinypascal.json",
        input_path: "input/tinypascal_large.txt",
        table_path: "table/tinypascal_ll1_table.csv",
        lr_table_path: "table/tinypascal_ll1_lr_table.csv",
        generate_table: true,
        parsers: GENERAL_PLUS_LLLR,
    },
    // GrammarConfig {
    //     name: "json_tokenized",
    //     grammar_path: "grammars/json_tokenized.json",
    //     input_path: "input/json_tokenized_large.txt",
    //     table_path: "table/json_tokenized_glr_table.csv",
    //     lr_table_path: "table/json_tokenized_lr_table.csv",
    //     generate_table: true,
    // }
    // GrammarConfig {
    //     name: "json_large",
    //     grammar_path: "grammars/json.json",
    //     input_path: "input/json_medium.txt",
    //     table_path: "table/json_glr_table.csv",
    //     lr_table_path: "table/json_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig {
    //     name: "sexp_large",
    //     grammar_path: "grammars/sexp.json",
    //     input_path: "input/sexp_large.txt",
    //     table_path: "table/sexp_glr_table.csv",
    //     lr_table_path: "table/sexp_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig {
    //     name: "calc_large",
    //     grammar_path: "grammars/calc.json",
    //     input_path: "input/calc_large.txt",
    //     table_path: "table/calc_glr_table.csv",
    //     lr_table_path: "table/calc_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig {
    //     name: "tinyc_large",
    //     grammar_path: "grammars/tinyc.json",
    //     input_path: "input/tinyc_large.txt",
    //     table_path: "table/tinyc_glr_table.csv",
    //     lr_table_path: "table/tinyc_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig {
    //     name: "css_small",
    //     grammar_path: "grammars/css.json",
    //     input_path: "input/css_small.txt",
    //     table_path: "table/css_glr_table.csv",
    //     lr_table_path: "table/css_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig {
    //     name: "ansi_c_large",
    //     grammar_path: "grammars/ansi_c.json",
    //     input_path: "input/ansi_c_large.txt",
    //     table_path: "table/ansi_c_glr_table.csv",
    //     lr_table_path: "table/ansi_c_lr_table.csv",
    //     generate_table: false,
    // },
    // GrammarConfig {
    //     name: "pascal_large",
    //     grammar_path: "grammars/pascal.json",
    //     input_path: "input/pascal_large.txt",
    //     table_path: "table/pascal_glr_table.csv",
    //     lr_table_path: "table/pascal_lr_table.csv",
    //     generate_table: false,
    // },
    // GrammarConfig {
    //     name: "java_large",
    //     grammar_path: "grammars/jsl18.json",
    //     input_path: "input/java_large.txt",
    //     table_path: "table/java_glr_table.csv",
    //     lr_table_path: "table/java_lr_table.csv",
    //     generate_table: false,
    // },
    // GrammarConfig {
    //     name: "cpp_large",
    //     grammar_path: "grammars/cpp.json",
    //     input_path: "input/cpp_large.txt",
    //     table_path: "table/cpp_glr_table.csv",
    //     lr_table_path: "table/cpp_lr_table.csv",
    //     generate_table: false,
    // },
];

const WARMUP_ITERATIONS: u32 = 1;
const MIN_ITERATIONS: u32 = 5;
const MAX_ITERATIONS: u32 = 20;
const TARGET_TIME: Duration = Duration::from_millis(500);

#[derive(Clone)]
struct BenchmarkResult {
    parser: String,
    input_length: usize,
    token_count: usize,
    median_time_ns: f64,
    mad_ns: f64,
    peak_memory_bytes: usize,
    iterations: u32,
    success: bool,
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{},{}",
            self.parser,
            self.input_length,
            self.token_count,
            self.median_time_ns,
            self.mad_ns,
            self.peak_memory_bytes,
            self.iterations,
            self.success
        )
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

fn check_success<F>(mut parse_fn: F) -> bool
where
    F: FnMut() -> Option<ParseTree>,
{
    parse_fn().is_some()
}

// ============================================================================
// Main Benchmark Logic
// ============================================================================

fn run_benchmarks(config: &GrammarConfig) -> std::io::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Benchmarking: {} grammar", config.name);
    println!("  Grammar: {}", config.grammar_path);
    println!("  Input:   {}", config.input_path);
    println!("{}", "=".repeat(60));

    // Setup CSV file and write header
    fs::create_dir_all("results")?;
    let filename = format!("results/benchmark_{}.csv", config.name);
    let mut csv_file = File::create(&filename)?;
    writeln!(
        csv_file,
        "parser,input_length,token_count,median_time_ns,mad_ns,peak_memory_bytes,iterations,success"
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
    let mut earley = earley::LeoParser::new(grammar.clone());
    let ll_parser = if config.parsers.contains(&"LL") {
        Some(ll::LLParser::new(&grammar))
    } else {
        None
    };
    let lr_parser = if config.parsers.contains(&"LR") {
        Some(
            lr::LRParser::from_csv(config.lr_table_path, &grammar)
                .expect("Failed to load LR table"),
        )
    } else {
        None
    };

    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    // Load inputs
    let input_content = fs::read_to_string(config.input_path).expect("Failed to read input file");

    let mut lines: Vec<&str> = input_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    // Sort by length for nicer plotting
    lines.sort_by_key(|l| l.len());

    println!(
        "Found {} inputs (lengths: {} to {} bytes)",
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
        let cnf_tokens = cnf_grammar.tokenize(line).unwrap_or_default();
        let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

        println!(
            "\n  Input #{}: {} bytes, {} tokens",
            idx + 1,
            input_len,
            token_count
        );
        println!("{}", &line[..std::cmp::min(50, line.len())].trim());
        println!("{}", &line[..std::cmp::min(50, line.len())].trim());
        for parser_name in config.parsers {
            let result = match *parser_name {
                "Earley" => {
                    let success = check_success(|| earley.parse(tokens.clone()));
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
                        success,
                    }
                }
                "GLL" => {
                    let success = check_success(|| gll_parser.parse(&tokens));
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
                        success,
                    }
                }
                "RNGLR" => {
                    let success = check_success(|| rnglr.parse(&glr_tokens));
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
                        success,
                    }
                }
                "BRNGLR" => {
                    let success = check_success(|| brnglr.parse(&glr_tokens));
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
                        success,
                    }
                }
                "CYK" => {
                    let success = check_success(|| cyk::parse(&cnf_grammar, &cnf_tokens));
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
                        success,
                    }
                }
                "Valiant" => {
                    let success = check_success(|| valiant::parse(&cnf_grammar, &cnf_tokens));
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
                        success,
                    }
                }
                "LR" => {
                    if let Some(parser) = &lr_parser {
                        let success = check_success(|| parser.parse(&glr_tokens));
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
                            success,
                        }
                    } else {
                        panic!("LR parser configured but not loaded");
                    }
                }
                "LL" => {
                    if let Some(parser) = &ll_parser {
                        let success = check_success(|| parser.parse(&tokens));
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
                            success,
                        }
                    } else {
                        panic!("LL parser configured but not loaded");
                    }
                }
                _ => continue,
            };

            let status = if result.success { "✓" } else { "✗" };
            println!(
                "    [{}] {:8}: {:>12.0} ns ± {:>8.0} ns ({} iters)",
                status, result.parser, result.median_time_ns, result.mad_ns, result.iterations
            );

            // Write result to CSV immediately
            writeln!(csv_file, "{}", result.to_csv_row())?;
            csv_file.flush()?; // Ensure data is written to disk immediately
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
