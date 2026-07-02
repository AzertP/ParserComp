using System;
using System.Linq;

namespace ITP1_10_C
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                int n = int.Parse(Console.ReadLine());
                if (n == 0) break;
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                double a = 0; double ave = x.Sum(); ave /= n;
                for (int i = 0; i < n; i++)
                {
                    a += Math.Pow(ave - x[i],2);
                }
                a /= n;
                Console.WriteLine(Math.Sqrt(a));
            }
        }
    }
}
