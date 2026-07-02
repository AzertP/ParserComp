using System;
using System.Collections.Generic;

class Program {
  static void Main(string[] args) {
    int kase = 1;
    while(true) {
      int x = int.Parse(Console.ReadLine());
      if (x == 0) break;
      Console.WriteLine("Case {0}: {1}", kase, x);
      kase++;
    }
  }
}
