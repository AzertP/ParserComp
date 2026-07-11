use tree_sitter_language::LanguageFn;

extern "C" {
    fn tree_sitter_gamma3() -> *const ();
}

pub const LANGUAGE: LanguageFn = unsafe { LanguageFn::from_raw(tree_sitter_gamma3) };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_small_accepted_inputs_without_errors() {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&LANGUAGE.into())
            .expect("Error loading gamma3 parser");

        for source in ["b", "bb", "bbb"] {
            let tree = parser.parse(source, None).expect("parser returned no tree");
            assert!(!tree.root_node().has_error(), "failed to parse {source}");
        }
    }
}
