using System;
using System.Linq;
using System.Collections;
class Program {
    static void Main() {
        int[] input = Console.ReadLine().Split().Select(int.Parse).ToArray();
        int W = input[0], H = input[1], x = input[2], y = input[3], r = input[4];
        bool ok = true;
        ok &= 0 <= x - r && x + r <= W;
        ok &= 0 <= y - r && y + r <= H;
        Console.WriteLine(ok ? "Yes" : "No");
    }
}

