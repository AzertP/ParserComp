module.exports = grammar({
  name: 'gamma2',

  extras: $ => [/\s/],

  conflicts: $ => [
    [$.s, $.b_chain],
    [$.s],
    [$.b_chain],
  ],

  rules: {
    source_file: $ => $.s,

    s: $ => choice(
      seq(optional($.b_chain), optional($.b_chain), $.s, 'a'),
      seq('b', 'b', 'b'),
    ),

    b_chain: $ => choice(
      seq('b', 'b'),
      seq('b', 'b', $.b_chain),
    ),
  },
});
