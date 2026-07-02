using System;
using System.Linq;
using System.Collections;
using System.Diagnostics;
class Program {
    static void Main() {
        int N = int.Parse(Console.ReadLine());
        int[] A = Console.ReadLine().Split().Select(int.Parse).ToArray();
        Console.WriteLine("{0} {1} {2}", A.Min(), A.Max(), A.Select(x => (long)x).Sum());
    }
}

