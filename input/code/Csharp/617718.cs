using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  enum Kind {
    S, H, C, D
  }

  static void Main(string[] args) {

    var sc = new Scanner();
    int N = sc.NextInt();
    bool[,] have = new bool[4, 13];
    for(int i = 0; i < N; i++) {
      Kind k;
      string s = sc.Next();
      if(s == "S") k = Kind.S;
      else if(s == "H") k = Kind.H;
      else if(s == "C") k = Kind.C;
      else k = Kind.D;
      int j = sc.NextInt();
      j--;
      have[(int)k, j] = true;
    }

    for(int i = 0; i < 13; i++) {
      if(have[0, i] == false) Console.WriteLine("S " + (i + 1));
    }
    for(int i = 0; i < 13; i++) {
      if(have[1, i] == false) Console.WriteLine("H " + (i + 1));
    }
    for(int i = 0; i < 13; i++) {
      if(have[2, i] == false) Console.WriteLine("C " + (i + 1));
    }
    for(int i = 0; i < 13; i++) {
      if(have[3, i] == false) Console.WriteLine("D " + (i + 1));
    }
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
