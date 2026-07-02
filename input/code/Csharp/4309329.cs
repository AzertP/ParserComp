using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                var line = Console.ReadLine().Split(' ');
                var m = int.Parse(line[0]);
                var f = int.Parse(line[1]);
                var r = int.Parse(line[2]);

                if (m == -1 && f == -1 && r == -1)
                {
                    break;
                }

                if (m == -1 || f == -1)
                {
                    Console.WriteLine("F");
                }
                else if (m + f >= 80)
                {
                    Console.WriteLine("A");
                }
                else if (m + f >= 65)
                {
                    Console.WriteLine("B");
                }
                else if (m + f >= 50)
                {
                    Console.WriteLine("C");
                }
                else if (m + f >= 30)
                {
                    if (r >= 50)
                    {
                        Console.WriteLine("C");
                    }
                    else
                    {
                        Console.WriteLine("D");
                    }
                }
                else
                {
                    Console.WriteLine("F");
                }
            }
        }
    }
}

