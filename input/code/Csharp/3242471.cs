using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static double Minkowski(int[] x, int[] y, double p) {
    double sum = 0;
    for (int i = 0; i < x.Length; i++) {
      sum += Math.Pow(Math.Abs(x[i] - y[i]), p);
    }
    return Math.Pow(sum, 1.0 / p);
  }
  static IEnumerable<int> Dist(int[] x, int[] y) {
    for (int i = 0; i < x.Length; i++) {
      yield return Math.Abs(x[i] - y[i]);
    }
  }
  static void Main() {
    int n = int.Parse(Console.ReadLine());
    var x = Console.ReadLine().Split().Select(int.Parse).ToArray();
    var y = Console.ReadLine().Split().Select(int.Parse).ToArray();
    Console.WriteLine(Minkowski(x, y, 1));
    Console.WriteLine(Minkowski(x, y, 2));
    Console.WriteLine(Minkowski(x, y, 3));
    Console.WriteLine(Dist(x, y).Max());
  }
}
