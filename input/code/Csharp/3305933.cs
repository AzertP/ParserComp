using System;
using System.Linq;

class Program {
  static bool Hoge(int[] w, int k, int p) {
    int t = 0;
    for (int i = 0; i < w.Length; i++) {
      if (k < 0) return false;
      else if (w[i] <= p - t) {
        t += w[i];
        if (t == p) { k--; t = 0; }
      } else { k--; t = 0; i--; }
    }
    return true;
  }

  static void Main() {
    var a = Console.ReadLine().Split().Select(int.Parse).ToArray();
    int n = a[0];
    int k = a[1];
    var w = Enumerable.Range(1, n).Select(_ => int.Parse(Console.ReadLine())).ToArray();
    int L = 0;
    int r = w.Sum();
    while (L < r) {
      int m = (L + r) / 2;
      if (Hoge(w, k - 1, m)) r = m;
      else L = m + 1;
    }
    Console.WriteLine(L);
  }
}
