using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication59
{
    class Program
    {
        static void Main()
        {
            Console.ReadLine();
            int[] A = Console.ReadLine().Split().Select(int.Parse).ToArray();
            Console.ReadLine();
            int[] B = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int S = 0;
            for(int C = 0; C < B.Length; C++)
            {
                if (B[C] > A[A.Length / 2])
                {
                    for (int D = A.Length-1; D>-1; D--)
                    {
                        if (B[C] == A[D]) { S++; break; }
                    }
                }
                else
                {
                    for (int D = 0; D < A.Length; D++)
                    {
                        if (B[C] == A[D]) { S++; break; }
                    }
                }
            }
            Console.WriteLine(S);
        }
    }
}
