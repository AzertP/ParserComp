//! Validate extracted OpenJDK JavacParserTest snippets with Java lexer + generalized parsers.
//!
//! Runs Leo, GLL, RNGLR, and BRNGLR over the same token stream and records
//! accept/reject/lexing outcomes for each parser.
//!
//! Usage:
//!   cargo run --release --bin validate_javac_lex_generalized

use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parsers::{earley_leo, gll, glr};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];

#[derive(Debug)]
struct CaseMeta {
    rel_path: String,
    method: String,
    source_line: String,
    has_expected_diagnostics: String,
    expected_diagnostics: String,
    has_erroneous_tree_check: String,
}

fn read_manifest(path: &Path) -> Result<Vec<CaseMeta>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {:?}: {}", path, e))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("empty manifest: {:?}", path))?
        .map_err(|e| format!("failed to read manifest header: {}", e))?;
    let columns: Vec<&str> = header.split('\t').collect();

    let index = |name: &str| -> Result<usize, String> {
        columns
            .iter()
            .position(|col| *col == name)
            .ok_or_else(|| format!("missing manifest column {:?}", name))
    };

    let file_idx = index("file")?;
    let method_idx = index("method")?;
    let source_line_idx = index("source_line")?;
    let has_diag_idx = index("has_expected_diagnostics")?;
    let diag_idx = index("expected_diagnostics")?;
    let has_err_idx = index("has_erroneous_tree_check")?;

    let mut cases = Vec::new();
    for line in lines {
        let line = line.map_err(|e| format!("failed to read manifest row: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        cases.push(CaseMeta {
            rel_path: fields.get(file_idx).unwrap_or(&"").to_string(),
            method: fields.get(method_idx).unwrap_or(&"").to_string(),
            source_line: fields.get(source_line_idx).unwrap_or(&"").to_string(),
            has_expected_diagnostics: fields.get(has_diag_idx).unwrap_or(&"").to_string(),
            expected_diagnostics: fields.get(diag_idx).unwrap_or(&"").to_string(),
            has_erroneous_tree_check: fields.get(has_err_idx).unwrap_or(&"").to_string(),
        });
    }

    Ok(cases)
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

fn write_row(
    out: &mut File,
    case: &CaseMeta,
    parser: &str,
    bytes: usize,
    chars: usize,
    lexed_tokens: usize,
    encoded_tokens: usize,
    status: &str,
    accepted: bool,
    elapsed_ms: f64,
    note: &str,
) -> std::io::Result<()> {
    writeln!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}",
        escape_tsv(&case.rel_path),
        escape_tsv(&case.method),
        escape_tsv(&case.source_line),
        escape_tsv(&case.has_expected_diagnostics),
        escape_tsv(&case.expected_diagnostics),
        escape_tsv(&case.has_erroneous_tree_check),
        bytes,
        chars,
        lexed_tokens,
        encoded_tokens,
        parser,
        status,
        accepted,
        elapsed_ms,
        escape_tsv(note),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let cases_root = PathBuf::from(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("test/openjdk-test"),
    );
    let manifest_path = PathBuf::from(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("test/openjdk-test/manifest.tsv"),
    );
    let out_path = PathBuf::from(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("test/openjdk-test/validate_java_result.tsv"),
    );

    let grammar = grammars::load_grammar_from_file("grammars/java_tok.json")
        .map_err(|e| format!("failed to load tokenized Java grammar: {}", e))?;
    let lexer = Lexer::from_file("grammars/lexer/java_regex.json")
        .map_err(|e| format!("failed to load Java lexer: {}", e))?;
    let mut rnglr = glr::RnglrParser::import_table_from_csv("table/java_tok_glr_table.csv")
        .map_err(|e| format!("failed to load RNGLR table: {}", e))?;
    let mut brnglr = glr::BrnglrParser::import_table_from_csv("table/java_tok_glr_table.csv")
        .map_err(|e| format!("failed to load BRNGLR table: {}", e))?;
    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    let cases = read_manifest(&manifest_path)?;

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut out = File::create(&out_path)?;
    writeln!(
        out,
        "file\tmethod\tsource_line\thas_expected_diagnostics\texpected_diagnostics\thas_erroneous_tree_check\tbytes\tchars\tlexed_tokens\tencoded_tokens\tparser\tstatus\taccepted\telapsed_ms\tnote"
    )?;

    println!(
        "Running lexical validation over {} cases with grammar {}",
        cases.len(),
        grammar.name
    );

    for (idx, case) in cases.iter().enumerate() {
        let path = cases_root.join(&case.rel_path);
        let content = fs::read_to_string(&path)?;
        let bytes = content.len();
        let chars = content.chars().count();

        let lexed = match lexer.lex(&content) {
            Ok(lexed) => lexed,
            Err(e) => {
                for parser in PARSERS {
                    write_row(
                        &mut out, case, parser, bytes, chars, 0, 0, "LEX_FAIL", false, 0.0, &e,
                    )?;
                }
                out.flush()?;
                println!(
                    "[{:>3}/{}] LEX_FAIL   all parsers  {}",
                    idx + 1,
                    cases.len(),
                    case.rel_path
                );
                continue;
            }
        };

        let ids = match encode(&lexed, &grammar) {
            Ok(ids) => ids,
            Err(e) => {
                for parser in PARSERS {
                    write_row(
                        &mut out,
                        case,
                        parser,
                        bytes,
                        chars,
                        lexed.len(),
                        0,
                        "ENCODE_FAIL",
                        false,
                        0.0,
                        &e,
                    )?;
                }
                out.flush()?;
                println!(
                    "[{:>3}/{}] ENCODE_FAIL all parsers  {}",
                    idx + 1,
                    cases.len(),
                    case.rel_path
                );
                continue;
            }
        };

        let glr_ids: Vec<i32> = ids.iter().map(|id| (*id + 1) as i32).collect();
        let mut statuses = Vec::new();

        for parser in PARSERS {
            let start = Instant::now();
            let result = match *parser {
                "Leo" => {
                    let mut leo = earley_leo::LeoParser::new(grammar.clone());
                    catch_unwind(AssertUnwindSafe(|| leo.parse(ids.clone())))
                }
                "GLL" => {
                    let mut gll_parser = gll::GLLParser::new(&grammar);
                    catch_unwind(AssertUnwindSafe(|| gll_parser.parse(&ids)))
                }
                "RNGLR" => catch_unwind(AssertUnwindSafe(|| rnglr.parse(&glr_ids))),
                "BRNGLR" => catch_unwind(AssertUnwindSafe(|| brnglr.parse(&glr_ids))),
                _ => unreachable!(),
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (status, accepted, note) = match result {
                Ok(Some(_)) => ("ACCEPT", true, String::new()),
                Ok(None) => ("REJECT", false, String::new()),
                Err(_) => ("PANIC", false, "panic during parse".to_string()),
            };

            statuses.push(format!("{}={}", parser, status));
            write_row(
                &mut out,
                case,
                parser,
                bytes,
                chars,
                lexed.len(),
                ids.len(),
                status,
                accepted,
                elapsed_ms,
                &note,
            )?;
        }
        out.flush()?;

        println!(
            "[{:>3}/{}] {}  {}",
            idx + 1,
            cases.len(),
            statuses.join(" "),
            case.rel_path
        );
    }

    println!();
    println!("Results written to {}", out_path.display());
    Ok(())
}
