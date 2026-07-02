using System;
using System.Linq;

namespace _2_A
{
    class Program
    {
        static void Main(string[] args)
        {
            int N = int.Parse(Console.ReadLine());
            int[] A = Console.ReadLine().Split().Select(int.Parse).ToArray();
            bool flag = true;
            int p = 0;
            while (flag)
            {
                flag = false;
                for (int j = N - 1; j > 0; j--)
                {
                    if (A[j] < A[j - 1])
                    {
                        int x = A[j];
                        A[j] = A[j - 1];
                        A[j - 1] = x;
                        flag = true;
                        p++;
                    }
                }
            }
            Console.Write(A[0]);
            for (int y = 1; y < N; y++)
            {
                Console.Write(" " + A[y]);
            }
            Console.WriteLine();
            Console.WriteLine(p);
        }
    }
}
