module.exports = grammar({
  name: 'gamma3',

  extras: $ => [/\s/],

  conflicts: $ => [
    [$.s],
  ],

  rules: {
    source_file: $ => $.s,

    s: $ => choice(
      seq($.s, $.s, $.s),
      seq($.s, $.s),
      'b',
    ),
  },
});
