using System;
using System.Linq;
using System.Collections.Generic;
namespace _5_B
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> H = new List<int>();
            List<int> W = new List<int>();
            while (true)
            {
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                if (x[0] == 0 && x[1] == 0)
                {
                    break;
                }
                else
                {
                    H.Add(x[0]);
                    W.Add(x[1]);
                }
            }
            for (int X = 0; X < H.Count; X++)
            {
                for (int i = 0; i < W[X]; i++)
                {
                    Console.Write("#");
                }
                Console.WriteLine();
                for (int i = 1; i < H[X] - 1; i++)
                {
                    Console.Write("#");
                    for (int I = 1; I < W[X] - 1; I++)
                    {
                        Console.Write(".");
                    }
                    Console.WriteLine("#");
                }
                for (int i = 0; i < W[X]; i++)
                {
                    Console.Write("#");
                }
                Console.WriteLine();
                Console.WriteLine();
            }
        }
    }
}
