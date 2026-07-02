using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static void Main(string[] args) {
    Solve();
  }

  static void Solve() {
    string[] inp = Console.ReadLine().Split(' ');
    int H = int.Parse(inp[0]);
    int W = int.Parse(inp[1]);
    if (H + W == 0) return;

    for(int y = 0; y < H; y++) {
      for(int x = 0; x < W; x++) {
        Console.Write((x + y) % 2 == 0 ? '#' : '.');
      }
      Console.WriteLine();
    }
    Console.WriteLine();

    Solve();
  }
}
