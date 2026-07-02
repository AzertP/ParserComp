using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static void SelectionSort(List<KeyValuePair<char, char>> a, int n) {
    for (int i = 0; i < n; i++) {
      int minj = i;
      for (int j = i; j < n; j++) {
        if (a[j].Value < a[minj].Value) minj = j;
      }
      if (i != minj) {
        var tmp = a[i];
        a[i] = a[minj];
        a[minj] = tmp;
      }
    }
  }
  static bool IsStable(List<KeyValuePair<char, char>> a, List<KeyValuePair<char, char>> b, int n) {
    for (int i = 0; i < n; i++) {
      if (a[i].Key != b[i].Key) return false;
    }
    return true;
  }
  static void Print(List<KeyValuePair<char, char>> c, int n) {
    for (int i = 0; i < n - 1; i++) {
      Console.Write("{0}{1} ", c[i].Key, c[i].Value);
    }
    Console.WriteLine("{0}{1}", c[n - 1].Key, c[n - 1].Value);
  }
  static void Main() {
    int n = int.Parse(Console.ReadLine());
    var s = Console.ReadLine().Split();
    var a = new List<KeyValuePair<char, char>>();
    var b = new List<KeyValuePair<char, char>>();
    foreach (var c in s) {
      var t = new KeyValuePair<char, char>(c[0], c[1]);
      a.Add(t);
      b.Add(t);
    }
    a = a.OrderBy(x => x.Value).ToList();
    Print(a, n);
    Console.WriteLine("Stable");
    SelectionSort(b, n);
    Print(b, n);
    Console.WriteLine(IsStable(a, b, n) ? "Stable" : "Not stable");
  }
}
