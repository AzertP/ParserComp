using System;
using System.Linq;

namespace ITP1_7_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int[,] a = new int[x[0],x[1]]; int[,] b = new int[x[1],x[2]];
            for (int i = 0; i < x[0]; i++)
            {
                int[] n = Console.ReadLine().Split().Select(int.Parse).ToArray();
                for (int j = 0; j < x[1]; j++)
                {
                    a[i,j] = n[j];
                }
            }
            for (int i = 0; i < x[1]; i++)
            {
                int[] n = Console.ReadLine().Split().Select(int.Parse).ToArray();
                for (int j = 0; j < x[2]; j++)
                {
                    b[i,j] = n[j];
                }
            }
            for (int i = 0; i < x[0]; i++)
            {
                long n = 0;
                for (int j = 0; j < x[2] - 1; j++)
                {
                    n = 0;
                    for (int k = 0; k < x[1]; k++)
                    {
                        n += a[i,k] * b[k,j];
                    }
                    Console.Write(n + " ");
                }
                n = 0;
                for (int k = 0; k < x[1]; k++)
                {
                    n += a[i,k] * b[k,x[2] - 1];
                }
                Console.WriteLine(n);
            }
        }
    }
}
