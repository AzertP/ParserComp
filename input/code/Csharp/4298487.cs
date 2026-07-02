using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var n = int.Parse(Console.ReadLine());

            for (var i = 1; i <= n; i++)
            {
                if (i % 3 == 0)
                {
                    Console.Write($" {i}");
                }
                else
                {
                    var x = i;
                    while (x > 0)
                    {
                        if (x % 10 == 3)
                        {
                            Console.Write($" {i}");
                            break;
                        }
                        x /= 10;
                    }
                }
            }
            Console.WriteLine();
        }
    }
}

