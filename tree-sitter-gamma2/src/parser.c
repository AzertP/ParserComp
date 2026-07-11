#include "tree_sitter/parser.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#define LANGUAGE_VERSION 14
#define STATE_COUNT 17
#define LARGE_STATE_COUNT 2
#define SYMBOL_COUNT 6
#define ALIAS_COUNT 0
#define TOKEN_COUNT 3
#define EXTERNAL_TOKEN_COUNT 0
#define FIELD_COUNT 0
#define MAX_ALIAS_SEQUENCE_LENGTH 4
#define PRODUCTION_ID_COUNT 1

enum ts_symbol_identifiers {
  anon_sym_a = 1,
  anon_sym_b = 2,
  sym_source_file = 3,
  sym_s = 4,
  sym_b_chain = 5,
};

static const char * const ts_symbol_names[] = {
  [ts_builtin_sym_end] = "end",
  [anon_sym_a] = "a",
  [anon_sym_b] = "b",
  [sym_source_file] = "source_file",
  [sym_s] = "s",
  [sym_b_chain] = "b_chain",
};

static const TSSymbol ts_symbol_map[] = {
  [ts_builtin_sym_end] = ts_builtin_sym_end,
  [anon_sym_a] = anon_sym_a,
  [anon_sym_b] = anon_sym_b,
  [sym_source_file] = sym_source_file,
  [sym_s] = sym_s,
  [sym_b_chain] = sym_b_chain,
};

static const TSSymbolMetadata ts_symbol_metadata[] = {
  [ts_builtin_sym_end] = {
    .visible = false,
    .named = true,
  },
  [anon_sym_a] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_b] = {
    .visible = true,
    .named = false,
  },
  [sym_source_file] = {
    .visible = true,
    .named = true,
  },
  [sym_s] = {
    .visible = true,
    .named = true,
  },
  [sym_b_chain] = {
    .visible = true,
    .named = true,
  },
};

static const TSSymbol ts_alias_sequences[PRODUCTION_ID_COUNT][MAX_ALIAS_SEQUENCE_LENGTH] = {
  [0] = {0},
};

static const uint16_t ts_non_terminal_alias_map[] = {
  0,
};

static const TSStateId ts_primary_state_ids[STATE_COUNT] = {
  [0] = 0,
  [1] = 1,
  [2] = 2,
  [3] = 3,
  [4] = 4,
  [5] = 5,
  [6] = 6,
  [7] = 7,
  [8] = 8,
  [9] = 9,
  [10] = 10,
  [11] = 11,
  [12] = 12,
  [13] = 13,
  [14] = 14,
  [15] = 15,
  [16] = 16,
};

static bool ts_lex(TSLexer *lexer, TSStateId state) {
  START_LEXER();
  eof = lexer->eof(lexer);
  switch (state) {
    case 0:
      if (eof) ADVANCE(1);
      if (lookahead == 'a') ADVANCE(2);
      if (lookahead == 'b') ADVANCE(3);
      if (('\t' <= lookahead && lookahead <= '\r') ||
          lookahead == ' ') SKIP(0);
      END_STATE();
    case 1:
      ACCEPT_TOKEN(ts_builtin_sym_end);
      END_STATE();
    case 2:
      ACCEPT_TOKEN(anon_sym_a);
      END_STATE();
    case 3:
      ACCEPT_TOKEN(anon_sym_b);
      END_STATE();
    default:
      return false;
  }
}

static const TSLexMode ts_lex_modes[STATE_COUNT] = {
  [0] = {.lex_state = 0},
  [1] = {.lex_state = 0},
  [2] = {.lex_state = 0},
  [3] = {.lex_state = 0},
  [4] = {.lex_state = 0},
  [5] = {.lex_state = 0},
  [6] = {.lex_state = 0},
  [7] = {.lex_state = 0},
  [8] = {.lex_state = 0},
  [9] = {.lex_state = 0},
  [10] = {.lex_state = 0},
  [11] = {.lex_state = 0},
  [12] = {.lex_state = 0},
  [13] = {.lex_state = 0},
  [14] = {.lex_state = 0},
  [15] = {.lex_state = 0},
  [16] = {.lex_state = 0},
};

static const uint16_t ts_parse_table[LARGE_STATE_COUNT][SYMBOL_COUNT] = {
  [0] = {
    [ts_builtin_sym_end] = ACTIONS(1),
    [anon_sym_a] = ACTIONS(1),
    [anon_sym_b] = ACTIONS(1),
  },
  [1] = {
    [sym_source_file] = STATE(12),
    [sym_s] = STATE(5),
    [sym_b_chain] = STATE(2),
    [anon_sym_b] = ACTIONS(3),
  },
};

static const uint16_t ts_small_parse_table[] = {
  [0] = 3,
    ACTIONS(3), 1,
      anon_sym_b,
    STATE(3), 1,
      sym_b_chain,
    STATE(13), 1,
      sym_s,
  [10] = 3,
    ACTIONS(3), 1,
      anon_sym_b,
    STATE(3), 1,
      sym_b_chain,
    STATE(15), 1,
      sym_s,
  [20] = 2,
    ACTIONS(7), 1,
      anon_sym_b,
    ACTIONS(5), 2,
      ts_builtin_sym_end,
      anon_sym_a,
  [28] = 2,
    ACTIONS(9), 1,
      ts_builtin_sym_end,
    ACTIONS(11), 1,
      anon_sym_a,
  [35] = 2,
    ACTIONS(13), 1,
      anon_sym_b,
    STATE(14), 1,
      sym_b_chain,
  [42] = 1,
    ACTIONS(16), 2,
      ts_builtin_sym_end,
      anon_sym_a,
  [47] = 2,
    ACTIONS(5), 1,
      ts_builtin_sym_end,
    ACTIONS(18), 1,
      anon_sym_a,
  [54] = 2,
    ACTIONS(21), 1,
      anon_sym_b,
    STATE(14), 1,
      sym_b_chain,
  [61] = 2,
    ACTIONS(24), 1,
      ts_builtin_sym_end,
    ACTIONS(26), 1,
      anon_sym_a,
  [68] = 1,
    ACTIONS(30), 1,
      anon_sym_b,
  [72] = 1,
    ACTIONS(32), 1,
      ts_builtin_sym_end,
  [76] = 1,
    ACTIONS(34), 1,
      anon_sym_a,
  [80] = 1,
    ACTIONS(36), 1,
      anon_sym_b,
  [84] = 1,
    ACTIONS(38), 1,
      anon_sym_a,
  [88] = 1,
    ACTIONS(7), 1,
      anon_sym_b,
};

static const uint32_t ts_small_parse_table_map[] = {
  [SMALL_STATE(2)] = 0,
  [SMALL_STATE(3)] = 10,
  [SMALL_STATE(4)] = 20,
  [SMALL_STATE(5)] = 28,
  [SMALL_STATE(6)] = 35,
  [SMALL_STATE(7)] = 42,
  [SMALL_STATE(8)] = 47,
  [SMALL_STATE(9)] = 54,
  [SMALL_STATE(10)] = 61,
  [SMALL_STATE(11)] = 68,
  [SMALL_STATE(12)] = 72,
  [SMALL_STATE(13)] = 76,
  [SMALL_STATE(14)] = 80,
  [SMALL_STATE(15)] = 84,
  [SMALL_STATE(16)] = 88,
};

static const TSParseActionEntry ts_parse_actions[] = {
  [0] = {.entry = {.count = 0, .reusable = false}},
  [1] = {.entry = {.count = 1, .reusable = false}}, RECOVER(),
  [3] = {.entry = {.count = 1, .reusable = true}}, SHIFT(11),
  [5] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_s, 3, 0, 0),
  [7] = {.entry = {.count = 1, .reusable = true}}, SHIFT(9),
  [9] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_source_file, 1, 0, 0),
  [11] = {.entry = {.count = 1, .reusable = true}}, SHIFT(7),
  [13] = {.entry = {.count = 2, .reusable = true}}, REDUCE(sym_b_chain, 2, 0, 0), SHIFT(4),
  [16] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_s, 2, 0, 0),
  [18] = {.entry = {.count = 2, .reusable = true}}, REDUCE(sym_s, 2, 0, 0), REDUCE(sym_s, 3, 0, 0),
  [21] = {.entry = {.count = 2, .reusable = true}}, REDUCE(sym_b_chain, 2, 0, 0), SHIFT(16),
  [24] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_s, 4, 0, 0),
  [26] = {.entry = {.count = 3, .reusable = true}}, REDUCE(sym_s, 2, 0, 0), REDUCE(sym_s, 3, 0, 0), REDUCE(sym_s, 4, 0, 0),
  [30] = {.entry = {.count = 1, .reusable = true}}, SHIFT(6),
  [32] = {.entry = {.count = 1, .reusable = true}},  ACCEPT_INPUT(),
  [34] = {.entry = {.count = 1, .reusable = true}}, SHIFT(8),
  [36] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_b_chain, 3, 0, 0),
  [38] = {.entry = {.count = 1, .reusable = true}}, SHIFT(10),
};

#ifdef __cplusplus
extern "C" {
#endif
#ifdef TREE_SITTER_HIDE_SYMBOLS
#define TS_PUBLIC
#elif defined(_WIN32)
#define TS_PUBLIC __declspec(dllexport)
#else
#define TS_PUBLIC __attribute__((visibility("default")))
#endif

TS_PUBLIC const TSLanguage *tree_sitter_gamma2(void) {
  static const TSLanguage language = {
    .version = LANGUAGE_VERSION,
    .symbol_count = SYMBOL_COUNT,
    .alias_count = ALIAS_COUNT,
    .token_count = TOKEN_COUNT,
    .external_token_count = EXTERNAL_TOKEN_COUNT,
    .state_count = STATE_COUNT,
    .large_state_count = LARGE_STATE_COUNT,
    .production_id_count = PRODUCTION_ID_COUNT,
    .field_count = FIELD_COUNT,
    .max_alias_sequence_length = MAX_ALIAS_SEQUENCE_LENGTH,
    .parse_table = &ts_parse_table[0][0],
    .small_parse_table = ts_small_parse_table,
    .small_parse_table_map = ts_small_parse_table_map,
    .parse_actions = ts_parse_actions,
    .symbol_names = ts_symbol_names,
    .symbol_metadata = ts_symbol_metadata,
    .public_symbol_map = ts_symbol_map,
    .alias_map = ts_non_terminal_alias_map,
    .alias_sequences = &ts_alias_sequences[0][0],
    .lex_modes = ts_lex_modes,
    .lex_fn = ts_lex,
    .primary_state_ids = ts_primary_state_ids,
  };
  return &language;
}
#ifdef __cplusplus
}
#endif
