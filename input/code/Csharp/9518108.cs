using System;
using System.Linq;
using System.Collections;
class Program {
    static void Main() {
        int[] input = Console.ReadLine().Split().Select(int.Parse).ToArray();
        int a = input[0], b = input[1];
        Console.WriteLine("{0} {1} {2:0.000000}", a / b, a % b, (decimal)a / b);
    }
}

