using System;
using System.Linq;

namespace _6_A
{
    class Program
    {
        static void Main(string[] args)
        {
            int kosuu = int.Parse(Console.ReadLine());
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            Array.Reverse(x);
            for (int i = 0; i < kosuu; i++)
            {
                if (!(i == 0))
                {
                    Console.Write(" ");
                }
                Console.Write(x[i]);
            }
            Console.WriteLine();
        }
    }
}
