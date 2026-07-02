using System;
using System.Linq;

namespace _1_B
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            if (x[0] < x[1])
            {
                Array.Reverse(x);
            }
            while (x[1] > 0)
            {
                int r = x[0] % x[1];
                x[0] = x[1];
                x[1] = r;
            }
            Console.WriteLine(x[0]);
        }
    }
}
