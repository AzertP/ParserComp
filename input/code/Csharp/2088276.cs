using System;
using System.Linq;

namespace _5_C
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
                        if ((I+i) % 2 == 0)
                        {
                            Console.Write("#");
                        }
                        else
                        {
                            Console.Write(".");
                        }
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
    }
}
