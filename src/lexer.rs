//! Regex-based lexer for tokenized C grammars.
//!
//! Reads a lexer spec from a JSON file (e.g. `grammars/lexer/c_regex.json`)
//! and tokenizes C source text, producing a token stream whose terminal names
//! match those used in tokenized grammar files such as `ansi_c_tok.json`.
//!
//! Token classification rules
//! ---------------------------
//!  - `ID_OR_KEYWORD` + alias word    →  the alias terminal (e.g. `"or"` → `"||"`)
//!  - `ID_OR_KEYWORD` + keyword word  →  the keyword string itself (e.g. `"int"`)
//!  - `ID_OR_KEYWORD` + non-keyword   →  configured identifier terminal, default `"ID"`
//!  - `OP`                            →  the matched operator text (e.g. `";"`, `"->"`)
//!  - `STRING`, `INTEGER`, `REAL`     →  the token-type name unchanged

use crate::grammars::NumericGrammar;
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

// ============================================================================
// Public types
// ============================================================================

/// A single token produced by the lexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Terminal name as it appears in the grammar (e.g. `"ID"`, `"int"`, `";"`)
    pub kind: String,
    /// The raw source text that was matched
    pub text: String,
    /// Byte offset of the token start in the original input
    pub offset: usize,
}

// ============================================================================
// JSON deserialization helper
// ============================================================================

#[derive(Deserialize)]
struct LexerSpecJson {
    skip: Vec<String>,
    /// Each entry is a two-element array `[kind, regex_pattern]`.
    tokens: Vec<Vec<String>>,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    aliases: HashMap<String, String>,
    #[serde(default)]
    identifier_kind: Option<String>,
}

// ============================================================================
// Lexer
// ============================================================================

/// A compiled lexer that tokenizes text according to a regex spec.
pub struct Lexer {
    skip_patterns: Vec<Regex>,
    /// Token patterns in priority order: `(kind, compiled_regex)`.
    token_patterns: Vec<(String, Regex)>,
    keywords: HashSet<String>,
    aliases: HashMap<String, String>,
    identifier_kind: String,
}

impl Lexer {
    /// Load and compile a [`Lexer`] from a JSON spec file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read lexer spec {:?}: {}", path.as_ref(), e))?;
        Self::from_str(&content)
    }

    /// Load and compile a [`Lexer`] from a JSON string.
    pub fn from_str(json: &str) -> Result<Self, String> {
        let spec: LexerSpecJson =
            serde_json::from_str(json).map_err(|e| format!("Failed to parse lexer JSON: {}", e))?;

        let skip_patterns = spec
            .skip
            .iter()
            .map(|pat| Regex::new(pat).map_err(|e| format!("Bad skip regex {:?}: {}", pat, e)))
            .collect::<Result<Vec<_>, _>>()?;

        let token_patterns = spec
            .tokens
            .iter()
            .map(|entry| {
                if entry.len() != 2 {
                    return Err(format!(
                        "Token entry must be [kind, regex], got {:?}",
                        entry
                    ));
                }
                let kind = entry[0].clone();
                let re = Regex::new(&entry[1])
                    .map_err(|e| format!("Bad token regex {:?}: {}", entry[1], e))?;
                Ok((kind, re))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let keywords: HashSet<String> = spec.keywords.into_iter().collect();

        Ok(Lexer {
            skip_patterns,
            token_patterns,
            keywords,
            aliases: spec.aliases,
            identifier_kind: spec.identifier_kind.unwrap_or_else(|| "ID".to_string()),
        })
    }

    /// Tokenize `input`, returning a [`Vec<Token>`] or an error describing
    /// where the input could not be matched.
    pub fn lex(&self, input: &str) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let mut pos = 0usize;

        while pos < input.len() {
            let remaining = &input[pos..];

            // 1. Skip whitespace / comments / preprocessor directives
            if let Some(skip_len) = self.match_skip(remaining) {
                pos += skip_len;
                continue;
            }

            // 2. Match the next token (longest-match via pattern priority)
            if let Some((raw_kind, match_len)) = self.match_token(remaining) {
                let text = remaining[..match_len].to_string();
                let kind = self.resolve_kind(raw_kind, &text);
                tokens.push(Token {
                    kind,
                    text,
                    offset: pos,
                });
                pos += match_len;
            } else {
                let snip_end = input.len().min(pos + 30);
                return Err(format!(
                    "Unexpected input at byte {}: {:?}",
                    pos,
                    &input[pos..snip_end]
                ));
            }
        }

        Ok(tokens)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Try each skip pattern; return the match length for the first one that
    /// anchors at position 0, or `None`.
    fn match_skip(&self, s: &str) -> Option<usize> {
        for re in &self.skip_patterns {
            if let Some(m) = re.find(s) {
                // All skip patterns start with `^`, so a match is always at 0.
                return Some(m.end());
            }
        }
        None
    }

    /// Try each token pattern in order; return `(kind, match_len)` for the
    /// first one that anchors at position 0, or `None`.
    fn match_token(&self, s: &str) -> Option<(String, usize)> {
        for (kind, re) in &self.token_patterns {
            if let Some(m) = re.find(s) {
                return Some((kind.clone(), m.end()));
            }
        }
        None
    }

    /// Map a raw token kind + matched text to the terminal name used by the
    /// grammar.
    fn resolve_kind(&self, raw_kind: String, text: &str) -> String {
        match raw_kind.as_str() {
            "ID_OR_KEYWORD" => {
                if let Some(kind) = self.aliases.get(text) {
                    kind.clone()
                } else if self.keywords.contains(text) {
                    text.to_string() // emit the keyword itself, e.g. "int"
                } else {
                    self.identifier_kind.clone()
                }
            }
            "OP" => text.to_string(), // emit the operator text, e.g. ";" or "->"
            _ => raw_kind,            // STRING, INTEGER, REAL — keep as-is
        }
    }
}

// ============================================================================
// Encoding
// ============================================================================

/// Encode a token stream to numeric terminal IDs using the grammar's terminal
/// table.
///
/// Returns `Err` if any token kind is absent from the grammar's terminal table.
/// This can happen for tokens that are lexically valid but not referenced by
/// the grammar (e.g. `TYPE_ID` / `ENUM_ID` require semantic analysis).
pub fn encode(tokens: &[Token], grammar: &NumericGrammar) -> Result<Vec<u32>, String> {
    tokens
        .iter()
        .map(|tok| {
            grammar.terminals.get_id(&tok.kind).ok_or_else(|| {
                format!(
                    "Token kind {:?} (text {:?} at byte {}) not in grammar terminal table",
                    tok.kind, tok.text, tok.offset
                )
            })
        })
        .collect()
}
