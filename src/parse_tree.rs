use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ParseSymbol {
    NonTerminal(String),
    Terminal(String),
}

impl fmt::Display for ParseSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseSymbol::NonTerminal(s) => write!(f, "{}", s),
            ParseSymbol::Terminal(s) => write!(f, "'{}'", s),
        }
    }
}

/// A parse tree node: (name, children)
/// Equivalent to Python's (name_string, list_of_children) tuple representation
#[derive(Debug, Clone, PartialEq)]
pub struct ParseTree {
    pub name: ParseSymbol,
    pub children: Vec<ParseTree>,
}

impl ParseTree {
    /// Create a new parse tree node
    pub fn new(name: ParseSymbol, children: Vec<ParseTree>) -> Self {
        ParseTree { name, children }
    }

    pub fn from_str(name: &str, children: Vec<ParseTree>) -> Self {
        ParseTree {
            name: ParseSymbol::NonTerminal(name.to_string()),
            children,
        }
    }

    /// Create a leaf node (no children)
    pub fn leaf(name: &str) -> Self {
        ParseTree {
            name: ParseSymbol::Terminal(name.to_string()),
            children: Vec::new(),
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Get the number of children
    pub fn num_children(&self) -> usize {
        self.children.len()
    }

    /// Pretty print the tree with indentation
    pub fn pretty_print(&self) -> String {
        self.pretty_print_indent(0)
    }

    fn pretty_print_indent(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        if self.children.is_empty() {
            format!("{}{}", prefix, self.name)
        } else {
            let children_str: Vec<String> = self
                .children
                .iter()
                .map(|c| c.pretty_print_indent(indent + 1))
                .collect();
            format!("{}({}\n{})", prefix, self.name, children_str.join("\n"))
        }
    }

    /// Display tree as ASCII art with box-drawing characters
    /// Output format:
    ///
    /// ```text
    /// <program>
    /// └─ <statement>
    ///     ├─ 'a'
    ///     └─ 'b'
    /// ```
    pub fn display(&self) -> String {
        let mut lines = Vec::new();
        self.build_display(&mut lines, String::new(), true, true);
        lines.join("\n")
    }

    fn build_display(&self, lines: &mut Vec<String>, prefix: String, is_last: bool, is_root: bool) {
        // Build current line
        if is_root {
            lines.push(self.name.to_string());
        } else {
            let connector = if is_last { "└─ " } else { "├─ " };
            lines.push(format!("{}{}{}", prefix, connector, self.name.to_string()));
        }

        // Build prefix for children
        let child_prefix = if is_root {
            String::new()
        } else if is_last {
            format!("{}    ", prefix)
        } else {
            format!("{}│   ", prefix)
        };

        // Process children
        let num_children = self.children.len();
        for (i, child) in self.children.iter().enumerate() {
            let is_last_child = i == num_children - 1;
            child.build_display(lines, child_prefix.clone(), is_last_child, false);
        }
    }

    /// Convert to Python tuple-style string representation
    /// Output: ('name', [('child1', []), ('child2', [])])
    pub fn to_tuple_string(&self) -> String {
        if self.children.is_empty() {
            format!("('{}', [])", self.name)
        } else {
            let children_str: Vec<String> =
                self.children.iter().map(|c| c.to_tuple_string()).collect();
            format!("('{}', [{}])", self.name, children_str.join(", "))
        }
    }
}

/// Macro for convenient tree construction (similar to Python tuple syntax)
/// Usage: tree!("S", [tree!("A"), tree!("B", [tree!("c")])])
#[macro_export]
macro_rules! tree {
    // Leaf node
    ($name:expr) => {
        ParseTree::leaf($name)
    };
    // Node with children
    ($name:expr, [$($child:expr),* $(,)?]) => {
        ParseTree::from_str($name, vec![$($child),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf() {
        let leaf = ParseTree::leaf("x");
        assert_eq!(leaf.name, ParseSymbol::Terminal("x".to_string()));
        assert!(leaf.is_leaf());
    }

    #[test]
    fn test_tree() {
        // Equivalent to Python: ("S", [("A", [("a", [])]), ("B", [("b", [])])])
        let tree = ParseTree::from_str(
            "S",
            vec![
                ParseTree::from_str("A", vec![ParseTree::leaf("a")]),
                ParseTree::from_str("B", vec![ParseTree::leaf("b")]),
            ],
        );
        assert_eq!(tree.name, ParseSymbol::NonTerminal("S".to_string()));
        assert_eq!(tree.num_children(), 2);
        assert_eq!(
            tree.children[0].name,
            ParseSymbol::NonTerminal("A".to_string())
        );
        assert_eq!(
            tree.children[1].name,
            ParseSymbol::NonTerminal("B".to_string())
        );
    }

    #[test]
    fn test_macro() {
        let tree = tree!("S", [tree!("A", [tree!("a")]), tree!("B", [tree!("b")])]);

        assert_eq!(tree.name, ParseSymbol::NonTerminal("S".to_string()));
        assert_eq!(tree.num_children(), 2);
    }

    #[test]
    fn test_display() {
        // Build the example tree: do;while(g);
        let tree = tree!(
            "<program>",
            [tree!(
                "<statement>",
                [
                    tree!("<do>", [tree!("d"), tree!("o")]),
                    tree!("<statement>", [tree!(";")]),
                    tree!(
                        "<while>",
                        [tree!("w"), tree!("h"), tree!("i"), tree!("l"), tree!("e")]
                    ),
                    tree!(
                        "<paren_expr>",
                        [
                            tree!("("),
                            tree!(
                                "<expr>",
                                [tree!(
                                    "<test>",
                                    [tree!(
                                        "<sum>",
                                        [tree!("<term>", [tree!("<id>", [tree!("g")])])]
                                    )]
                                )]
                            ),
                            tree!(")")
                        ]
                    ),
                    tree!(";")
                ]
            )]
        );

        println!("ASCII tree:\n{}", tree.display());
        println!("Tuple format:\n{}", tree.to_tuple_string());
    }
}
