using System;
using System.Linq;
using System.Collections;
class Program {
    static void Main() {
        int[] A = Console.ReadLine().Split().Select(int.Parse).ToArray();
        Array.Sort(A);
        System.Console.WriteLine(String.Join(' ', A));
    }
}

