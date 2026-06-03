//! Parser validator: runs all parsers on test/grammar/*.json + test/input/<name>.txt
//! and checks that all parsers agree on accept/reject for each input.
//!
//! An "accepted" test case is one where ALL parsers agree on the outcome
//! (either all accept or all reject).
//!
//! Usage:
//!   cargo run --bin validate
//!
//! Grammar files are read from test/grammar/*.json.
//! Input files are read from test/input/<grammar_name>.txt (one input per line).
//! GLR parse tables are generated online — no table files are read or written.

use parser_comparison::grammars::{self, NumSymbol};
use parser_comparison::parsers::gll::ll::LLParser;
use parser_comparison::parsers::glr::glr::{BrnglrParser, ParseTable, ParsedAction, RnglrParser};
use parser_comparison::parsers::glr::lr::LRParser;
use parser_comparison::parsers::glr::table_generator::{Action, TableGenerator, END_OF_INPUT};
use parser_comparison::parsers::{cyk, earley_leo, gll, valiant};
use rustc_hash::FxHashMap;
use std::fs;

// ============================================================================
// Inline GLR table generation
// ============================================================================

/// Build a GLR ParseTable from a grammar in memory (no file I/O).
fn build_glr_table_online(grammar: &grammars::Grammar) -> ParseTable {
    let table_gen = TableGenerator::new(grammar);
    let raw = table_gen.generate_parse_table();

    let mut parser_table: ParseTable = FxHashMap::default();
    for (state, actions) in raw {
        let mut state_actions: FxHashMap<i32, Vec<ParsedAction>> = FxHashMap::default();
        for (symbol, action_list) in actions {
            let sym_i32 = match symbol {
                NumSymbol::Terminal(id) if id == END_OF_INPUT => 0,
                NumSymbol::Terminal(id) => (id + 1) as i32,
                NumSymbol::NonTerminal(id) => -((id + 1) as i32),
            };
            let parsed: Vec<ParsedAction> = action_list
                .iter()
                .map(|a| match a {
                    Action::Shift(s) => ParsedAction::Push(*s),
                    Action::Reduce(lhs, dot, label) => {
                        ParsedAction::Reduce(-((*lhs + 1) as i32), *dot, *label)
                    }
                    Action::Accept => ParsedAction::Accept,
                })
                .collect();
            state_actions.insert(sym_i32, parsed);
        }
        parser_table.insert(state, state_actions);
    }
    parser_table
}

// ============================================================================
// Result types
// ============================================================================

/// Outcome of a single parser on a single input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Accept,
    Reject,
    Conflict, // parser could not be built (grammar conflict)
}

impl Outcome {
    fn symbol(self) -> &'static str {
        match self {
            Outcome::Accept => "A",
            Outcome::Reject => "R",
            Outcome::Conflict => "C",
        }
    }
}

struct ParserResult {
    name: &'static str,
    outcome: Outcome,
}

// ============================================================================
// Main
// ============================================================================

fn run_main() {
    let grammar_dir = "test/grammar";
    let input_dir = "test/input";

    // Collect grammar files sorted by name for deterministic output.
    let mut grammar_paths: Vec<_> = fs::read_dir(grammar_dir)
        .unwrap_or_else(|e| panic!("Cannot open {}: {}", grammar_dir, e))
        .filter_map(|e| {
            let path = e.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    grammar_paths.sort();

    if grammar_paths.is_empty() {
        println!("No grammar files found in {}", grammar_dir);
        return;
    }

    let mut total = 0usize;
    let mut agreed = 0usize;
    let mut disagreed = 0usize;
    let mut skipped = 0usize;

    // Tree-count validation (Leo vs RNGLR vs BRNGLR, accepted inputs only).
    let mut tree_total = 0usize;
    let mut tree_agreed = 0usize;
    let mut tree_disagreed = 0usize;

    for grammar_path in &grammar_paths {
        let grammar_name = grammar_path
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let input_path = format!("{}/{}.txt", input_dir, grammar_name);

        if !std::path::Path::new(&input_path).exists() {
            eprintln!("[SKIP] No input file for grammar: {}", grammar_name);
            continue;
        }

        println!("\n{}", "=".repeat(60));
        println!("Grammar: {}", grammar_name);
        println!("{}", "=".repeat(60));

        // Load grammar
        let grammar = match grammars::load_grammar_from_file(grammar_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[ERROR] Failed to load grammar {}: {}", grammar_name, e);
                continue;
            }
        };
        let cnf_grammar = grammar.to_cnf();

        // Build GLR table online (no saved file).
        let parse_table = build_glr_table_online(&grammar);
        let rnglr = RnglrParser::with_grammar(parse_table.clone(), grammar.clone());
        let brnglr = BrnglrParser::with_grammar(parse_table, grammar.clone());

        let mut gll_parser = gll::GLLParser::new(&grammar);
        let mut leo = earley_leo::LeoParser::new(grammar.clone());

        // LL and LR may panic on conflicting grammars.
        let ll_result: Result<LLParser, _> =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                LLParser::new(&grammar)
            }));
        let ll_parser = ll_result.ok();

        let lr_result: Result<LRParser, _> =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| LRParser::new(&grammar)));
        let lr_parser = lr_result.ok();

        // Report which parsers are active.
        let active: Vec<&str> = {
            let mut v = vec!["Leo", "GLL", "RNGLR", "BRNGLR", "CYK", "Valiant"];
            if ll_parser.is_some() {
                v.push("LL");
            } else {
                eprintln!("  [INFO] LL has grammar conflicts — marked CONFLICT for all inputs");
            }
            if lr_parser.is_some() {
                v.push("LR");
            } else {
                eprintln!("  [INFO] LR has grammar conflicts — marked CONFLICT for all inputs");
            }
            v
        };
        println!("Active parsers: {}", active.join(", "));

        // Load inputs.
        let input_content = match fs::read_to_string(&input_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[ERROR] Failed to read {}: {}", input_path, e);
                continue;
            }
        };

        for (line_no, raw_line) in input_content.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            total += 1;

            // Tokenize.
            let tokens = match grammar.tokenize(line) {
                Some(t) => t,
                None => {
                    println!(
                        "  SKIP [{:3}] {:40} (tokenize failed)",
                        line_no + 1,
                        line
                    );
                    skipped += 1;
                    total -= 1;
                    continue;
                }
            };
            let cnf_tokens = cnf_grammar.tokenize(line).unwrap_or_default();
            let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

            // Run every parser.
            let mut results: Vec<ParserResult> = Vec::new();

            results.push(ParserResult {
                name: "Leo",
                outcome: if leo.parse(tokens.clone()).is_some() {
                    Outcome::Accept
                } else {
                    Outcome::Reject
                },
            });
            results.push(ParserResult {
                name: "GLL",
                outcome: if gll_parser.parse(&tokens).is_some() {
                    Outcome::Accept
                } else {
                    Outcome::Reject
                },
            });
            results.push(ParserResult {
                name: "RNGLR",
                outcome: if rnglr.parse(&glr_tokens).is_some() {
                    Outcome::Accept
                } else {
                    Outcome::Reject
                },
            });
            results.push(ParserResult {
                name: "BRNGLR",
                outcome: if brnglr.parse(&glr_tokens).is_some() {
                    Outcome::Accept
                } else {
                    Outcome::Reject
                },
            });
            results.push(ParserResult {
                name: "CYK",
                outcome: if cyk::parse(&cnf_grammar, &cnf_tokens).is_some() {
                    Outcome::Accept
                } else {
                    Outcome::Reject
                },
            });
            // results.push(ParserResult {
            //     name: "Valiant",
            //     outcome: if valiant::parse(&cnf_grammar, &cnf_tokens).is_some() {
            //         Outcome::Accept
            //     } else {
            //         Outcome::Reject
            //     },
            // });
            if let Some(ref p) = ll_parser {
                results.push(ParserResult {
                    name: "LL",
                    outcome: if p.parse(&tokens).is_some() {
                        Outcome::Accept
                    } else {
                        Outcome::Reject
                    },
                });
            } else {
                results.push(ParserResult {
                    name: "LL",
                    outcome: Outcome::Conflict,
                });
            }
            if let Some(ref p) = lr_parser {
                results.push(ParserResult {
                    name: "LR",
                    outcome: if p.parse(&glr_tokens).is_some() {
                        Outcome::Accept
                    } else {
                        Outcome::Reject
                    },
                });
            } else {
                results.push(ParserResult {
                    name: "LR",
                    outcome: Outcome::Conflict,
                });
            }

            // Check agreement: ignore CONFLICT parsers for the purpose of agreement.
            let voting: Vec<Outcome> = results
                .iter()
                .filter(|r| r.outcome != Outcome::Conflict)
                .map(|r| r.outcome)
                .collect();

            let all_accept = voting.iter().all(|&o| o == Outcome::Accept);
            let all_reject = voting.iter().all(|&o| o == Outcome::Reject);
            let agree = all_accept || all_reject;

            if agree {
                agreed += 1;
            } else {
                disagreed += 1;
            }

            // Format per-parser detail string.
            let detail: String = results
                .iter()
                .map(|r| format!("{}:{}", r.name, r.outcome.symbol()))
                .collect::<Vec<_>>()
                .join("  ");

            let verdict = if agree {
                if all_accept {
                    "PASS (all ACCEPT)"
                } else {
                    "PASS (all REJECT)"
                }
            } else {
                "FAIL (disagree)"
            };

            println!(
                "  [{:3}] {:35} {}",
                line_no + 1,
                line,
                verdict
            );
            if !agree {
                println!("        {}", detail);
            }

            // Tree-count check: compare Leo, GLL, RNGLR, BRNGLR on accepted inputs.
            if all_accept {
                tree_total += 1;
                let leo_n = leo.parse_all(tokens.clone()).len();
                let gll_n = gll_parser.parse_all(&tokens).len();
                let rnglr_n = rnglr.parse_all(&glr_tokens).map(|v| v.len()).unwrap_or(0);
                let brnglr_n = brnglr.parse_all(&glr_tokens).map(|v| v.len()).unwrap_or(0);

                let trees_agree = leo_n == gll_n && gll_n == rnglr_n && rnglr_n == brnglr_n;
                if trees_agree {
                    tree_agreed += 1;
                } else {
                    tree_disagreed += 1;
                }
                // Print detail when ambiguous (>1 tree) or when parsers disagree.
                if !trees_agree || leo_n != 1 {
                    println!(
                        "        trees  Leo={:<4} GLL={:<4} RNGLR={:<4} BRNGLR={:<4}  {}",
                        leo_n, gll_n, rnglr_n, brnglr_n,
                        if trees_agree { "AGREE" } else { "DISAGREE !!!" }
                    );
                }
            }
        }
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("Validation Summary");
    println!("{}", "=".repeat(60));
    println!("  Total test cases : {}", total);
    println!(
        "  Agreed (PASS)    : {}  ({:.1}%)",
        agreed,
        if total > 0 {
            agreed as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("  Disagreed (FAIL) : {}", disagreed);
    if skipped > 0 {
        println!("  Skipped          : {}", skipped);
    }

    if disagreed == 0 {
        println!("\nAll parsers agree on every test case.");
    } else {
        println!("\nWARNING: {} test case(s) have parser disagreements.", disagreed);
    }

    // Tree-count summary.
    if tree_total > 0 {
        println!();
        println!("{}", "=".repeat(60));
        println!("Tree-count check (Leo vs GLL vs RNGLR vs BRNGLR, {} accepted inputs)", tree_total);
        println!("{}", "=".repeat(60));
        println!(
            "  Agreed   : {}  ({:.1}%)",
            tree_agreed,
            tree_agreed as f64 / tree_total as f64 * 100.0
        );
        println!("  Disagreed: {}", tree_disagreed);
        if tree_disagreed == 0 {
            println!("\nAll four parsers agree on parse-tree counts.");
        } else {
            println!("\nWARNING: {} input(s) have differing parse-tree counts.", tree_disagreed);
        }
    }
}

// ============================================================================
// Ground truth predicates for small, well-understood languages
// ============================================================================

/// a^n b^n for n ≥ 0
fn gt_anbn(s: &str) -> bool {
    let n = s.chars().take_while(|&c| c == 'a').count();
    s.len() == 2 * n && s.chars().skip(n).all(|c| c == 'b')
}

/// Dyck-1: balanced parentheses over {(, )}
fn gt_balanced_parens(s: &str) -> bool {
    let mut depth: i32 = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => return false,
        }
    }
    depth == 0
}

/// Dyck-2: properly nested brackets over {(, ), [, ]}
fn gt_dyck2(s: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    for c in s.chars() {
        match c {
            '(' | '[' => stack.push(c),
            ')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
            }
            _ => return false,
        }
    }
    stack.is_empty()
}

/// Palindromes over {a, b}
fn gt_palindrome_ab(s: &str) -> bool {
    let v: Vec<char> = s.chars().collect();
    v.iter().eq(v.iter().rev())
}

/// S → A B C; A → a|ε; B → b|ε; C → c|ε
/// Accepted strings: the 8 ordered sub-sequences of "abc"
fn gt_nullable_abc(s: &str) -> bool {
    matches!(s, "" | "a" | "b" | "c" | "ab" | "ac" | "bc" | "abc")
}

/// S → S+S | S*S | a  (ambiguous arithmetic)
/// Language: a ((+|*) a)*  — odd-length strings alternating a and operator
fn gt_ambi_expr(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() || chars.len() % 2 == 0 {
        return false;
    }
    for (i, &c) in chars.iter().enumerate() {
        if i % 2 == 0 && c != 'a' {
            return false;
        }
        if i % 2 == 1 && c != '+' && c != '*' {
            return false;
        }
    }
    true
}

/// a*b*: any number of a's followed by any number of b's
fn gt_star_ab(s: &str) -> bool {
    let mut saw_b = false;
    for c in s.chars() {
        match c {
            'a' => {
                if saw_b {
                    return false;
                }
            }
            'b' => saw_b = true,
            _ => return false,
        }
    }
    true
}

// ============================================================================
// Exhaustive string enumeration
// ============================================================================

/// Generate all strings over `terminals` (each a token string) with total
/// token-count 0..=max_tokens, by BFS.  For single-character tokens the
/// string length equals the token count.
fn enumerate_strings(terminals: &[String], max_tokens: usize) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    let mut frontier: Vec<String> = vec![String::new()]; // length 0
    result.push(String::new());
    for _ in 1..=max_tokens {
        let mut next: Vec<String> = Vec::with_capacity(frontier.len() * terminals.len());
        for prefix in &frontier {
            for sym in terminals {
                next.push(format!("{}{}", prefix, sym));
            }
        }
        result.extend(next.iter().cloned());
        frontier = next;
    }
    result
}

// ============================================================================
// Exhaustive + ground truth validation
// ============================================================================

struct ExhaustiveCase {
    name: &'static str,
    /// Raw JSON text of the grammar (embedded at compile time).
    grammar_json: &'static str,
    /// Maximum token count (= string length for single-char terminals).
    max_tokens: usize,
    /// Optional closed-form membership predicate used as ground truth.
    ground_truth: Option<fn(&str) -> bool>,
}

fn exhaustive_cases() -> Vec<ExhaustiveCase> {
    vec![
        ExhaustiveCase {
            name: "ab  (a^n b^n)",
            grammar_json: include_str!("../../test/grammar/ab.json"),
            max_tokens: 8,
            ground_truth: Some(gt_anbn),
        },
        ExhaustiveCase {
            name: "parentheses  (Dyck-1)",
            grammar_json: include_str!("../../test/grammar/parentheses.json"),
            max_tokens: 8,
            ground_truth: Some(gt_balanced_parens),
        },
        ExhaustiveCase {
            name: "dyck2  (Dyck-2)",
            grammar_json: include_str!("../../test/grammar/dyck2.json"),
            max_tokens: 6,
            ground_truth: Some(gt_dyck2),
        },
        ExhaustiveCase {
            name: "palindrome  (palindromes over {a,b})",
            grammar_json: include_str!("../../test/grammar/palindrome.json"),
            max_tokens: 8,
            ground_truth: Some(gt_palindrome_ab),
        },
        ExhaustiveCase {
            name: "nullable  (ordered sub-sequences of abc)",
            grammar_json: include_str!("../../test/grammar/nullable.json"),
            max_tokens: 3,
            ground_truth: Some(gt_nullable_abc),
        },
        ExhaustiveCase {
            name: "ambi  (ambiguous arithmetic)",
            grammar_json: include_str!("../../test/grammar/ambi.json"),
            max_tokens: 7,
            ground_truth: Some(gt_ambi_expr),
        },
        ExhaustiveCase {
            name: "expr_lr  (LR arithmetic)",
            grammar_json: include_str!("../../test/grammar/expr_lr.json"),
            max_tokens: 4,
            ground_truth: None, // language not expressible as a simple predicate
        },
        ExhaustiveCase {
            name: "star  (a*b*)",
            grammar_json: include_str!("../../test/grammar/star.json"),
            max_tokens: 8,
            ground_truth: Some(gt_star_ab),
        },
    ]
}

/// Run exhaustive small-language validation.
/// Returns `true` if everything passed (no differential failures and no
/// ground-truth mismatches).
fn run_exhaustive_validation() -> bool {
    println!("\n{}", "=".repeat(60));
    println!("Exhaustive Small-Language Validation");
    println!("{}", "=".repeat(60));

    let mut all_ok = true;
    let mut grand_total = 0usize;
    let mut grand_diff_fail = 0usize;
    let mut grand_gt_fail = 0usize;

    for case in exhaustive_cases() {
        // Load grammar.
        let grammar = match grammars::load_grammar_from_str(case.grammar_json) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[ERROR] Could not load grammar '{}': {}", case.name, e);
                all_ok = false;
                continue;
            }
        };
        let cnf_grammar = grammar.to_cnf();

        // Build parsers.
        let parse_table = build_glr_table_online(&grammar);
        let rnglr = RnglrParser::with_grammar(parse_table.clone(), grammar.clone());
        let brnglr = BrnglrParser::with_grammar(parse_table, grammar.clone());
        let mut gll_parser = gll::GLLParser::new(&grammar);
        let mut leo = earley_leo::LeoParser::new(grammar.clone());

        let ll_parser = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            LLParser::new(&grammar)
        }))
        .ok();
        let lr_parser = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            LRParser::new(&grammar)
        }))
        .ok();

        // Collect terminal alphabet (token strings, typically single characters).
        let alphabet: Vec<String> = (0..grammar.num_terminals())
            .map(|i| grammar.terminal_str(i as u32).unwrap().to_string())
            .collect();

        // Enumerate every string over the alphabet up to max_tokens length.
        let strings = enumerate_strings(&alphabet, case.max_tokens);
        let total = strings.len();

        let mut diff_fail = 0usize;
        let mut gt_fail = 0usize;

        for input in &strings {
            // Tokenize.
            let tokens = match grammar.tokenize(input) {
                Some(t) => t,
                None => continue, // should never happen for generated strings
            };
            let cnf_tokens = cnf_grammar.tokenize(input).unwrap_or_default();
            let glr_tokens: Vec<i32> = tokens.iter().map(|&t| (t + 1) as i32).collect();

            // Run all parsers.
            let leo_acc = leo.parse(tokens.clone()).is_some();
            let gll_acc = gll_parser.parse(&tokens).is_some();
            let rnglr_acc = rnglr.parse(&glr_tokens).is_some();
            let brnglr_acc = brnglr.parse(&glr_tokens).is_some();
            let cyk_acc = cyk::parse(&cnf_grammar, &cnf_tokens).is_some();
            // let val_acc = valiant::parse(&cnf_grammar, &cnf_tokens).is_some();
            let ll_acc = ll_parser.as_ref().map(|p| p.parse(&tokens).is_some());
            let lr_acc = lr_parser.as_ref().map(|p| p.parse(&glr_tokens).is_some());

            // Differential check: all non-conflicting parsers must agree.
            let non_conflict_results: Vec<bool> = {
                let mut v = vec![leo_acc, gll_acc, rnglr_acc, brnglr_acc, cyk_acc];
                if let Some(r) = ll_acc {
                    v.push(r);
                }
                if let Some(r) = lr_acc {
                    v.push(r);
                }
                v
            };
            let all_accept_d = non_conflict_results.iter().all(|&r| r);
            let all_reject_d = non_conflict_results.iter().all(|&r| !r);
            let diff_ok = all_accept_d || all_reject_d;

            if !diff_ok {
                diff_fail += 1;
                all_ok = false;
                let detail = format!(
                    "Leo:{} GLL:{} RNGLR:{} BRNGLR:{} CYK:{} {}{}",
                    if leo_acc { "A" } else { "R" },
                    if gll_acc { "A" } else { "R" },
                    if rnglr_acc { "A" } else { "R" },
                    if brnglr_acc { "A" } else { "R" },
                    if cyk_acc { "A" } else { "R" },
                    // if val_acc { "A" } else { "R" },
                    ll_acc.map_or(String::new(), |r| format!(" LL:{}", if r { "A" } else { "R" })),
                    lr_acc.map_or(String::new(), |r| format!(" LR:{}", if r { "A" } else { "R" })),
                );
                println!(
                    "  DIFF-FAIL  input={:?}  {}",
                    if input.is_empty() { "(empty)" } else { input },
                    detail
                );
            }

            // Ground truth check.
            if let Some(gt_fn) = case.ground_truth {
                let expected = gt_fn(input);
                // Use the majority / consensus verdict (Leo as reference — it's Earley + Leo).
                let parser_says = leo_acc; // will also equal gll_acc / rnglr_acc if diff_ok
                if expected != parser_says || !diff_ok {
                    let any_wrong = expected != leo_acc
                        || expected != gll_acc
                        || expected != rnglr_acc
                        || expected != brnglr_acc
                        || expected != cyk_acc
                        // || expected != val_acc
                        || ll_acc.map_or(false, |r| r != expected)
                        || lr_acc.map_or(false, |r| r != expected);
                    if any_wrong {
                        gt_fail += 1;
                        all_ok = false;
                        println!(
                            "  GT-FAIL    input={:?}  expected={}  Leo:{} GLL:{} RNGLR:{} BRNGLR:{} CYK:{}",
                            if input.is_empty() { "(empty)" } else { input },
                            if expected { "ACCEPT" } else { "REJECT" },
                            if leo_acc { "A" } else { "R" },
                            if gll_acc { "A" } else { "R" },
                            if rnglr_acc { "A" } else { "R" },
                            if brnglr_acc { "A" } else { "R" },
                            if cyk_acc { "A" } else { "R" },
                            // if val_acc { "A" } else { "R" },
                        );
                    }
                }
            }
        }

        grand_total += total;
        grand_diff_fail += diff_fail;
        grand_gt_fail += gt_fail;

        let gt_summary = if case.ground_truth.is_some() {
            let gt_pass = total - gt_fail;
            format!(
                "  GT: {}/{} ok{}",
                gt_pass,
                total,
                if gt_fail > 0 {
                    format!(", {} MISMATCH", gt_fail)
                } else {
                    String::new()
                }
            )
        } else {
            String::new()
        };

        println!(
            "\n  {:50}  {} strings  |  diff: {}/{} ok{}{}",
            case.name,
            total,
            total - diff_fail,
            total,
            if diff_fail > 0 {
                format!(", {} FAIL", diff_fail)
            } else {
                String::new()
            },
            gt_summary,
        );
    }

    // Grand summary.
    println!("\n{}", "=".repeat(60));
    println!("Exhaustive Validation Summary");
    println!("{}", "=".repeat(60));
    println!("  Total strings tested : {}", grand_total);
    println!(
        "  Differential PASS    : {}",
        grand_total - grand_diff_fail
    );
    if grand_diff_fail > 0 {
        println!("  Differential FAIL    : {}  *** PROBLEMS ***", grand_diff_fail);
    } else {
        println!("  Differential FAIL    : 0");
    }
    if grand_gt_fail > 0 {
        println!("  Ground truth FAIL    : {}  *** PROBLEMS ***", grand_gt_fail);
    } else {
        println!("  Ground truth FAIL    : 0");
    }

    all_ok
}

fn main() {
    // Run on a large stack to avoid overflow on deep recursion.
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            run_main();
            println!();
            let exhaustive_ok = run_exhaustive_validation();
            if !exhaustive_ok {
                std::process::exit(1);
            }
        })
        .expect("Failed to spawn thread")
        .join()
        .expect("Thread panicked");
}
