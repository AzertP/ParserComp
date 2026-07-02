using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication14
{
   class Program
    {
        static void Main()
        {
            int a = int.Parse(Console.ReadLine()), b=0, c=0;
            for(int u = 0; u < a; u++)
            {
                string[] w = Console.ReadLine().Split().ToArray(); if (w[0] == w[1]) { b++;c++;continue; } string p = w[0];
                Array.Sort(w);if (p == w[0]) c += 3; else b += 3;
            }
            Console.WriteLine("{0} {1}",b,c);
        }
    }
}
