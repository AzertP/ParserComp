using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static int InsertionSort(List<int> a, int n, int g) {
    int cnt = 0;
    for (int i = g; i < n; i++) {
      int v = a[i];
      int j = i - g;
      while (j >= 0 && a[j] > v) {
        a[j + g] = a[j];
        j -= g;
        cnt++;
      }
      a[j + g] = v;
    }
    return cnt;
  }

  static void Main() {
    int n = int.Parse(Console.ReadLine());
    var a = Enumerable.Range(1, n).Select(_ => int.Parse(Console.ReadLine())).ToList();
    var g = new List<int>();
    for (int i = 1; i <= n; i = i * 3 + 1) {
      g.Add(i);
    }
    int cnt = 0;
    g = g.OrderByDescending(x => x).ToList();
    foreach (var i in g) {
      cnt += InsertionSort(a, n, i);
    }
    Console.WriteLine(g.Count);
    Console.WriteLine(string.Join(" ", g.Select(x => x.ToString())).ToArray());
    Console.WriteLine(cnt);
    a.ForEach(Console.WriteLine);
  }
}
