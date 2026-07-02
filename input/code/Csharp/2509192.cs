using System;
using System.Linq;

namespace _6_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int[,] gyoretu = new int[x[0], x[1]];
            for (int i = 0; i < x[0]; i++)
            {
                int[] a = Console.ReadLine().Split().Select(int.Parse).ToArray();
                for (int j = 0; j < x[1]; j++)
                {
                    gyoretu[i, j] = a[j];
                }
            }
            int[] bekutoru = new int[x[1]];
            for (int i = 0; i < x[1]; i++)
            {
                bekutoru[i] = int.Parse(Console.ReadLine());
            }
            for (int i = 0; i < x[0]; i++)
            {
                long ret = 0;
                for (int j = 0; j < x[1]; j++)
                {
                    ret += gyoretu[i, j] * bekutoru[j];
                }
                Console.WriteLine(ret);
            }
        }
    }
}
