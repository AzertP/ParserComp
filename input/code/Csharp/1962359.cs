using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication29
{
    class Program
    {
        static void Main()
        {
            int[] A = Console.ReadLine().Split().Select(int.Parse).ToArray();
            while (true)
            {
                Array.Sort(A);
                if (A[1] % A[0] == 0) break;
                int B = A[0];
                A[0] = A[1] - A[0];
                A[1] = B;
            }
            Console.WriteLine(A[0]);
        }
    }
}
