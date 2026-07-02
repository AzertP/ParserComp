using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static void Main(string[] args) {
    var sc = new Scanner();
    int N = sc.NextInt();
    int[] v = new int[N];
    foreach (int i in Enumerable.Range(0, N)) v[i] = sc.NextInt();
    for (int i = N - 1; i > 0; i--) Console.Write(v[i] + " ");
    Console.WriteLine(v[0]);
  }

  class Scanner {
    string[] inp;
    int ptr;

    public Scanner() {
      inp = new string[0];
      ptr = 0;
    }

    private void Fetch() {
      if (ptr >= inp.Length) {
        inp = Console.ReadLine().Split(' ');
        ptr = 0;
      }
    }

    public int NextInt() {
      Fetch();
      return int.Parse(inp[ptr++]);
    }

    public string Next() {
      Fetch();
      return inp[ptr++];
    }

    public double NextDouble() {
      Fetch();
      return double.Parse(inp[ptr++]);
    }
  }

}
