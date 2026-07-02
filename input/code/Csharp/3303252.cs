using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static void Main() {
    var s = Console.ReadLine();
    var d = new Stack<int>();
    var p = new Stack<KeyValuePair<int, int>>();
    int a = 0;
    for (int i = 0; i < s.Length; i++) {
      if (s[i] == '\\') {
        d.Push(i);
      } else if (s[i] == '/' && d.Count > 0) {
        int b = d.Pop();
        int t = i - b;
        a += t;
        while (p.Count > 0 && b < p.Peek().Key) {
          t += p.Pop().Value;
        }
        p.Push(new KeyValuePair<int, int>(b, t));
      }
    }
    Console.WriteLine(a);
    Console.Write(p.Count);
    foreach (var item in p.Select(x => x.Value).Reverse().Select(x => x.ToString()).ToArray()) {
      Console.Write(" " + item);
    }
    Console.WriteLine();
  }
}
