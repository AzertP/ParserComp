//! Validate extracted OpenJDK JavacParserTest snippets with tree-sitter,
//! Java lexer, and generalized parsers.
//!
//! Runs tree-sitter-java over each raw source, then runs Leo, GLL, RNGLR,
//! and BRNGLR over the same token stream and records accept/reject/lexing
//! outcomes for each parser.
//!
//! Usage:
//!   cargo run --release --bin validate_java

use parser_comparison::grammars;
use parser_comparison::lexer::{encode, Lexer};
use parser_comparison::parsers::{earley_leo, gll, glr};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tree_sitter::{Node, Parser};

const TREE_SITTER_PARSER: &str = "TreeSitterJava";
const RECOGNITION_ORACLE: &str = "tree_sitter_java";
const JAVA_TREE_LEXER_SPEC: &str = "grammars/lexer/java_tree_regex.json";
const JAVA_TREE_GRAMMAR: &str = "grammars/java_tree.json";
const JAVA_TREE_GLR_TABLE: &str = "table/java_tree_glr_table.csv";
const LOCAL_PARSERS: &[&str] = &["Leo", "GLL", "RNGLR", "BRNGLR"];

#[derive(Debug)]
struct CaseMeta {
    rel_path: String,
    method: String,
    source_line: String,
    has_expected_diagnostics: String,
    expected_diagnostics: String,
    has_erroneous_tree_check: String,
}

#[derive(Debug)]
struct TreeSitterResult {
    status: String,
    accepted: bool,
    error_nodes: usize,
    missing_nodes: usize,
    note: String,
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

fn run_tree_sitter(parser: &mut Parser, source: &str) -> TreeSitterResult {
    let Some(tree) = parser.parse(source, None) else {
        return TreeSitterResult {
            status: "PARSE_FAIL".to_string(),
            accepted: false,
            error_nodes: 0,
            missing_nodes: 0,
            note: "tree-sitter parser returned no tree".to_string(),
        };
    };

    let root = tree.root_node();
    let (error_nodes, missing_nodes) = count_error_nodes(root);
    let accepted = !root.has_error() && error_nodes == 0 && missing_nodes == 0;
    let status = if accepted { "ACCEPT" } else { "ERROR_TREE" }.to_string();
    let note = format!("errors={} missing={}", error_nodes, missing_nodes);

    TreeSitterResult {
        status,
        accepted,
        error_nodes,
        missing_nodes,
        note,
    }
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
    oracle: &TreeSitterResult,
    elapsed_ms: f64,
    note: &str,
) -> std::io::Result<()> {
    let matches_expected = (accepted == oracle.accepted).to_string();
    let oracle_error_count = oracle.error_nodes + oracle.missing_nodes;
    let oracle_error_detail = format!(
        "tree_sitter_errors={} missing={}",
        oracle.error_nodes, oracle.missing_nodes
    );
    writeln!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}",
        escape_tsv(&case.rel_path),
        escape_tsv(&case.method),
        escape_tsv(&case.source_line),
        escape_tsv(&case.has_expected_diagnostics),
        escape_tsv(&case.expected_diagnostics),
        escape_tsv(&case.has_erroneous_tree_check),
        RECOGNITION_ORACLE,
        oracle.accepted,
        oracle_error_count,
        escape_tsv(&oracle_error_detail),
        bytes,
        chars,
        lexed_tokens,
        encoded_tokens,
        parser,
        status,
        accepted,
        matches_expected,
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

    let grammar = grammars::load_grammar_from_file(JAVA_TREE_GRAMMAR)
        .map_err(|e| format!("failed to load Java tree grammar: {}", e))?;
    let lexer = Lexer::from_file(JAVA_TREE_LEXER_SPEC)
        .map_err(|e| format!("failed to load Java tree lexer: {}", e))?;
    let mut rnglr = glr::RnglrParser::import_table_from_csv(JAVA_TREE_GLR_TABLE)
        .map_err(|e| format!("failed to load RNGLR table: {}", e))?;
    let mut brnglr = glr::BrnglrParser::import_table_from_csv(JAVA_TREE_GLR_TABLE)
        .map_err(|e| format!("failed to load BRNGLR table: {}", e))?;
    rnglr.set_grammar(grammar.clone());
    brnglr.set_grammar(grammar.clone());

    let mut tree_sitter_parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
    tree_sitter_parser
        .set_language(&language)
        .map_err(|e| format!("failed to load tree-sitter Java grammar: {}", e))?;

    let cases = read_manifest(&manifest_path)?;

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut out = File::create(&out_path)?;
    writeln!(
        out,
        "file\tmethod\tsource_line\thas_expected_diagnostics\texpected_diagnostics\thas_erroneous_tree_check\trecognition_oracle\texpected_accept\toracle_error_count\toracle_error_detail\tbytes\tchars\tlexed_tokens\tencoded_tokens\tparser\tstatus\taccepted\tmatches_expected\telapsed_ms\tnote"
    )?;

    println!(
        "Running tree-sitter oracle validation over {} cases with grammar {}",
        cases.len(),
        grammar.name
    );

    for (idx, case) in cases.iter().enumerate() {
        let path = cases_root.join(&case.rel_path);
        let content = fs::read_to_string(&path)?;
        let bytes = content.len();
        let chars = content.chars().count();

        let start = Instant::now();
        let tree_sitter_result = run_tree_sitter(&mut tree_sitter_parser, &content);
        let tree_sitter_elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        write_row(
            &mut out,
            case,
            TREE_SITTER_PARSER,
            bytes,
            chars,
            0,
            0,
            &tree_sitter_result.status,
            tree_sitter_result.accepted,
            &tree_sitter_result,
            tree_sitter_elapsed_ms,
            &tree_sitter_result.note,
        )?;

        let lexed = match lexer.lex(&content) {
            Ok(lexed) => lexed,
            Err(e) => {
                for parser in LOCAL_PARSERS {
                    write_row(
                        &mut out,
                        case,
                        parser,
                        bytes,
                        chars,
                        0,
                        0,
                        "LEX_FAIL",
                        false,
                        &tree_sitter_result,
                        0.0,
                        &e,
                    )?;
                }
                out.flush()?;
                println!(
                    "[{:>3}/{}] {} LEX_FAIL local parsers  {}",
                    idx + 1,
                    cases.len(),
                    format!("{}={}", TREE_SITTER_PARSER, tree_sitter_result.status),
                    case.rel_path
                );
                continue;
            }
        };

        let ids = match encode(&lexed, &grammar) {
            Ok(ids) => ids,
            Err(e) => {
                for parser in LOCAL_PARSERS {
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
                        &tree_sitter_result,
                        0.0,
                        &e,
                    )?;
                }
                out.flush()?;
                println!(
                    "[{:>3}/{}] {} ENCODE_FAIL local parsers  {}",
                    idx + 1,
                    cases.len(),
                    format!("{}={}", TREE_SITTER_PARSER, tree_sitter_result.status),
                    case.rel_path
                );
                continue;
            }
        };

        let glr_ids: Vec<i32> = ids.iter().map(|id| (*id + 1) as i32).collect();
        let mut statuses = vec![format!(
            "{}={}",
            TREE_SITTER_PARSER, tree_sitter_result.status
        )];

        for parser in LOCAL_PARSERS {
            let start = Instant::now();
            let result = match *parser {
                "Leo" => {
                    let mut leo = earley_leo::LeoParser::new(grammar.clone());
                    catch_unwind(AssertUnwindSafe(|| leo.parse(ids.clone()).is_some()))
                }
                "GLL" => {
                    let mut gll_parser = gll::GLLParser::new(&grammar);
                    catch_unwind(AssertUnwindSafe(|| gll_parser.parse_one(&ids).is_some()))
                }
                "RNGLR" => catch_unwind(AssertUnwindSafe(|| rnglr.parse(&glr_ids).is_some())),
                "BRNGLR" => catch_unwind(AssertUnwindSafe(|| brnglr.parse(&glr_ids).is_some())),
                _ => unreachable!(),
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (status, accepted, note) = match result {
                Ok(true) => ("ACCEPT", true, String::new()),
                Ok(false) => ("REJECT", false, String::new()),
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
                &tree_sitter_result,
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
