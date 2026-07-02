using System;
using System.Linq;

namespace _7_C
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int[,] a = new int[x[0] + 1, x[1] + 1];
            // r???c????????????
            for (int i = 0; i < x[0]; i++)
            {
                int[] now = Console.ReadLine().Split().Select(int.Parse).ToArray();
                for (int j = 0; j < x[1]; j++)
                {
                    a[i, j] = now[j];
                }
                a[i, x[1]] = now.Sum();
                Console.Write(a[i, 0]);
                for (int j = 1; j <= x[1]; j++)
                {
                    Console.Write(" " + a[i, j]);
                }
                Console.WriteLine();
            }
            int sum = 0;
            for (int i = 0; i < x[0]; i++)
            {
                sum += a[i, 0];
            }
            Console.Write(sum);
            for (int j = 1; j <= x[1]; j++)
            {
                sum = 0;
                for (int i = 0; i < x[0]; i++)
                {
                    sum += a[i, j];
                }
                Console.Write(" " + sum);
            }
            Console.WriteLine();
        }
    }
}
