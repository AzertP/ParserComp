//! Benchmarking tool that outputs CSV data for plotting parsing time vs input length.
//!
//! Usage:
//!   cargo run --release --bin benchmark_csv
//!
//! Output:
//!   Creates a CSV file in results/ with columns:
//!   parser, input_length, tokens, median_time_ns, mad_ns, iterations
use parser_comparison::grammars;
use parser_comparison::parse_tree::ParseTree;
use parser_comparison::parsers::glr::table_generator;
use parser_comparison::parsers::{cyk, earley, gll, glr, valiant};
use parser_comparison::parsers::gll::ll;
use parser_comparison::parsers::glr::lr;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
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
}

const CONFIGS: &[GrammarConfig] = &[
    // GrammarConfig{
    //     name: "lr_json_large",
    //     grammar_path: "grammars/lr_json.json",
    //     input_path: "input/json_large.txt",
    //     table_path: "table/lr_json_glr_table.csv",
    //     lr_table_path: "table/lr_json_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig{
    //     name: "lr_tinyc_large",
    //     grammar_path: "grammars/lr_tinyc.json",
    //     input_path: "input/tinyc_lr_large.txt",
    //     table_path: "table/lr_tinyc_glr_table.csv",
    //     lr_table_path: "table/lr_tinyc_lr_table.csv",
    //     generate_table: true,
    // },
    GrammarConfig{
        name: "lr_lua_large",
        grammar_path: "grammars/lr_lua.json",
        input_path: "input/lua_lr_large.txt",
        table_path: "table/lr_lua_glr_table.csv",
        lr_table_path: "table/lr_lua_lr_table.csv",
        generate_table: true,
    },



    // GrammarConfig{
    //     name: "json_with_lllr_large",
    //     grammar_path: "grammars/ll1_json.json",
    //     input_path: "input/json_ultra.txt",
    //     table_path: "table/json_ll1_table.csv",
    //     lr_table_path: "table/json_ll1_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig{
    //     name: "json_tokenized_with_lllr_large",
    //     grammar_path: "grammars/json_tokenized.json",
    //     input_path: "input/json_tokenized_large.txt",
    //     table_path: "table/json_tokenized_table.csv",
    //     lr_table_path: "table/json_tokenized_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig{
    //     name: "calc_with_lllr_large",
    //     grammar_path: "grammars/ll1_calc.json",
    //     input_path: "input/calc_large.txt",
    //     table_path: "table/calc_ll1_table.csv",
    //     lr_table_path: "table/calc_ll1_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig{
    //     name: "sexp_with_lllr_large",
    //     grammar_path: "grammars/ll1_sexp.json",
    //     input_path: "input/sexp_ll1.txt",
    //     table_path: "table/sexp_ll1_table.csv",
    //     lr_table_path: "table/sexp_ll1_lr_table.csv",
    //     generate_table: true,
    // },
    // GrammarConfig {
    //     name: "tinypascal_with_lllr_large",
    //     grammar_path: "grammars/ll1_tinypascal.json",
    //     input_path: "input/tinypascal_large.txt",
    //     table_path: "table/tinypascal_ll1_table.csv",
    //     lr_table_path: "table/tinypascal_ll1_lr_table.csv",
    //     generate_table: true,
    // }
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
    // Add more configurations as needed:
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

/// Number of warmup iterations before measuring
const WARMUP_ITERATIONS: u32 = 1;

/// Maximum iterations per measurement
const MAX_ITERATIONS: u32 = 3;

/// Parsers to benchmark (comment out to disable)
const PARSERS: &[&str] = &[
    "Earley", 
    "GLL", 
    "RNGLR", "BRNGLR", 
    // "LL", 
    "LR",
    // "CYK",      // Uncomment for CNF parsers (slower)
    // "Valiant",  // Uncomment for CNF parsers (slower)
];

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
    iterations: u32,
    success: bool,
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{}",
            self.parser,
            self.input_length,
            self.token_count,
            self.median_time_ns,
            self.mad_ns,
            self.iterations,
            self.success
        )
    }
}

// ============================================================================
// Measurement Functions
// ============================================================================

/// Measure a parsing function with statistical rigor
fn measure<F>(mut parse_fn: F) -> (f64, f64, u32)
where
    F: FnMut() -> Option<ParseTree>,
{
    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let result = black_box(parse_fn());
        // if let Some(tree) = result {
        //     println!("{}", tree.display());
        // }
    }

    // Determine iteration count based on time
    let mut times: Vec<f64> = Vec::new();
    let mut iterations = 0u32;

    while iterations < MAX_ITERATIONS {
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

    // Calculate MAD (Median Absolute Deviation) for robust spread measure
    let mut deviations: Vec<f64> = times.iter().map(|t| (t - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if deviations.len() % 2 == 0 {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    (median, mad, iterations)
}

/// Check if a parser succeeded (single run for validation)
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
        "parser,input_length,token_count,median_time_ns,mad_ns,iterations,success"
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
    // let ll_parser = ll::LLParser::new(&grammar);
    let lr_parser = lr::LRParser::from_csv(config.lr_table_path, &grammar)
        .expect("Failed to load LR table");

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
        for parser_name in PARSERS {
            let result = match *parser_name {
                "Earley" => {
                    let success = check_success(|| earley.parse(tokens.clone()));
                    let (median, mad, iters) = measure(|| earley.parse(tokens.clone()));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                "GLL" => {
                    let success = check_success(|| gll_parser.parse(&tokens));
                    let (median, mad, iters) = measure(|| gll_parser.parse(&tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                "RNGLR" => {
                    let success = check_success(|| rnglr.parse(&glr_tokens));
                    let (median, mad, iters) = measure(|| rnglr.parse(&glr_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                "BRNGLR" => {
                    let success = check_success(|| brnglr.parse(&glr_tokens));
                    let (median, mad, iters) = measure(|| brnglr.parse(&glr_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                "CYK" => {
                    let success = check_success(|| cyk::parse(&cnf_grammar, &cnf_tokens));
                    let (median, mad, iters) = measure(|| cyk::parse(&cnf_grammar, &cnf_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                "Valiant" => {
                    let success = check_success(|| valiant::parse(&cnf_grammar, &cnf_tokens));
                    let (median, mad, iters) =
                        measure(|| valiant::parse(&cnf_grammar, &cnf_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                "LR" => {
                    let success = check_success(|| lr_parser.parse(&glr_tokens));
                    let (median, mad, iters) = measure(|| lr_parser.parse(&glr_tokens));
                    BenchmarkResult {
                        parser: parser_name.to_string(),
                        input_length: input_len,
                        token_count,
                        median_time_ns: median,
                        mad_ns: mad,
                        iterations: iters,
                        success,
                    }
                }
                // "LL" => {
                //     let success = check_success(|| ll_parser.parse(&tokens));
                //     let (median, mad, iters) = measure(|| ll_parser.parse(&tokens));
                //     BenchmarkResult {
                //         parser: parser_name.to_string(),
                //         input_length: input_len,
                //         token_count,
                //         median_time_ns: median,
                //         mad_ns: mad,
                //         iterations: iters,
                //         success,
                //     }
                // }
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
    println!("  Max iterations: {}", MAX_ITERATIONS);
    println!("  Parsers: {:?}", PARSERS);

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
