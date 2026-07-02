using System;
using System.Linq;
using System.Collections;
class Program {
    static void Main() {
        int[] input = Console.ReadLine().Split().Select(int.Parse).ToArray();
        int a = input[0], b = input[1], c = input[2];
        int ans = 0;
        for (int i = a ; i <= b ; i++) {
            if (c % i == 0) ans++;
        }
        Console.WriteLine(ans);
    }
}

