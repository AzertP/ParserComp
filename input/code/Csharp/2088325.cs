using System;
using System.Linq;

namespace _5_A
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                if (x[0] == 0 && x[1] == 0)
                {
                    break;
                }
                for (int i = 0; i < x[0]; i++)
                {
                    for (int I = 0; I < x[1]; I++)
                    {
                        Console.Write("#");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
    }
}
