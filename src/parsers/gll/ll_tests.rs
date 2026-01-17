use super::*;
use crate::grammars::{NumSymbol, NumericGrammar};

fn create_test_grammar() -> NumericGrammar {
    // S -> 'a' S | 'b'
    let mut grammar = NumericGrammar::new("TestGrammar");

    // Add symbols
    let s_id = grammar.non_terminals.get_or_insert("S");
    let a_id = grammar.terminals.get_or_insert("a");
    let b_id = grammar.terminals.get_or_insert("b");

    grammar.start = s_id;

    // S -> 'a' S
    let prod1 = vec![NumSymbol::Terminal(a_id), NumSymbol::NonTerminal(s_id)];
    // S -> 'b'
    let prod2 = vec![NumSymbol::Terminal(b_id)];

    grammar.rules.insert(s_id, vec![prod1, prod2]);

    grammar
}

#[test]
fn test_ll_simple_accept() {
    let grammar = create_test_grammar();
    let parser = LLParser::new(&grammar);

    let a_id = grammar.terminals.get_id("a").unwrap();
    let b_id = grammar.terminals.get_id("b").unwrap();

    // "a a b"
    let input = vec![a_id, a_id, b_id];
    assert!(parser.recognize(&input));
}

#[test]
fn test_ll_simple_reject() {
    let grammar = create_test_grammar();
    let parser = LLParser::new(&grammar);

    let a_id = grammar.terminals.get_id("a").unwrap();
    let b_id = grammar.terminals.get_id("b").unwrap();

    // "a a" (incomplete)
    assert!(!parser.recognize(&vec![a_id, a_id]));

    // "b a" (wrong order)
    assert!(!parser.recognize(&vec![b_id, a_id]));
}

#[test]
#[should_panic(expected = "LL(1) Conflict")]
fn test_ll_conflict_panic() {
    // S -> 'a' | 'a'
    let mut grammar = NumericGrammar::new("Ambiguous");
    let s_id = grammar.non_terminals.get_or_insert("S");
    let a_id = grammar.terminals.get_or_insert("a");
    grammar.start = s_id;

    let prod1 = vec![NumSymbol::Terminal(a_id)];
    let prod2 = vec![NumSymbol::Terminal(a_id)];

    grammar.rules.insert(s_id, vec![prod1, prod2]);

    let _parser = LLParser::new(&grammar);
}

#[test]
fn test_ll_parse_tree() {
    use crate::parse_tree::ParseSymbol;

    let grammar = create_test_grammar();
    let parser = LLParser::new(&grammar);

    let a_id = grammar.terminals.get_id("a").unwrap();
    let b_id = grammar.terminals.get_id("b").unwrap();

    // "a b" -> S(a, S(b))
    let input = vec![a_id, b_id];
    let tree = parser.parse(&input).expect("Parse failed");

    println!("{:}", tree.display());
    assert_eq!(tree.name, ParseSymbol::NonTerminal("S".to_string()));
    assert_eq!(tree.children.len(), 2);
    assert_eq!(
        tree.children[0].name,
        ParseSymbol::Terminal("a".to_string())
    );

    let child_s = &tree.children[1];
    assert_eq!(child_s.name, ParseSymbol::NonTerminal("S".to_string()));
    assert_eq!(child_s.children.len(), 1);
    assert_eq!(
        child_s.children[0].name,
        ParseSymbol::Terminal("b".to_string())
    );
}
