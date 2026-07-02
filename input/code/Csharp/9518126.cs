using System;
using System.Linq;
using System.Collections;
using System.Diagnostics;
class Program {
    static string[] readLine() {
        return Console.ReadLine().Split();
    }
    static void Main() {
        while (true) {
            int[] A = readLine().Select(int.Parse).ToArray();
            int H = A[0], W = A[1];
            if (H == 0 && W == 0) break;
            for (int i = 0 ; i < H ; i++) {
                for (int j = 0 ; j < W ; j++) {
                    Console.Write("#");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}

