using System;
using System.Linq;
using System.Collections;
using System.Diagnostics;
using System.Collections.Generic;
class Program {
    static void Main() {
        int[,,] A = new int[4, 3, 10];
        for (int i = 0 ; i < 4 ; i++) {
            for (int j = 0 ; j < 3 ; j++) {
                for (int k = 0 ; k < 10 ; k++) {
                    A[i, j, k] = 0;
                }
            }
        }
        int N = int.Parse(Console.ReadLine());
        for (int i = 0 ; i < N ; i++) {
            int[] input = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int b = input[0], f = input[1], r = input[2], v = input[3];
            A[b - 1, f - 1, r - 1] += v; 
        }
        for (int i = 0 ; i < 4 ; i++) {
            for (int j = 0 ; j < 3 ; j++) {
                for (int k = 0 ; k < 10 ; k++) {
                    Console.Write(' ');
                    Console.Write(A[i, j, k]);
                }
                Console.Write('\n');
            }
            if (i + 1 < 4) {
                Console.WriteLine(String.Join("", Enumerable.Repeat("#", 20)));
            }
        }
    }
}

