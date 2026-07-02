using System;
using System.Linq;

namespace _2_B
{
    class Program
    {
        static void Main(string[] args)
        {
            int N = int.Parse(Console.ReadLine());
            int[] A = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int output = 0;
            for (int i = 0; i < N; i++)
            {
                int minj = i;
                for (int j = i; j < N; j++)
                {
                    if (A[j] < A[minj])
                    {
                        minj = j;
                    }
                }
                if (A[i] != A[minj])
                {
                    int x = A[i];
                    A[i] = A[minj];
                    A[minj] = x;
                    output++;
                }
            }
            Console.Write(A[0]);
            for (int i = 1; i < N; i++)
            {
                Console.Write(" " + A[i]);
            }
            Console.WriteLine();
            Console.WriteLine(output);
        }
    }
}
