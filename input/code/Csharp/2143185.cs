using System;
using System.Linq;

namespace _1_A
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = int.Parse(Console.ReadLine());
            int[] A = new int[n];
            A = Console.ReadLine().Split().Select(int.Parse).ToArray();
            Console.Write(A[0]);
            for (int i = 1; i < n; i++)
            {
                Console.Write(" " + A[i]);
            }
            Console.WriteLine();
            for (int i = 1; i < n; i++)
            {
                int v = A[i];
                int j = i - 1;
                while (j >= 0 && A[j] > v)
                {
                    A[j + 1] = A[j];
                    j--;
                }
                A[j + 1] = v;
                Console.Write(A[0]);
                for (int k = 1; k < n; k++)
                {
                    Console.Write(" " + A[k]);
                }
                Console.WriteLine();
            }
            Console.ReadLine();
        }
    }
}
