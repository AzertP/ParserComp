using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static void Main(string[] args) {
    var sc = new Scanner();
    int N = sc.NextInt();
    var list = new int[N];
    for(int i = 0; i < N; i++) list[i] = sc.NextInt();
    int cnt = 0;
    for(int i = 0; i < list.Length; i++) {
      for(int j = list.Length - 1; j > i; j--) {
        if(list[j] < list[j - 1]) {
          int t = list[j];
          list[j] = list[j - 1];
          list[j - 1] = t;
          cnt++;
        }
      }
    }
    Console.WriteLine(string.Join(" ", list.Select(x => x.ToString()).ToArray()));
    Console.WriteLine(cnt);
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
