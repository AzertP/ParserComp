using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication58
{
    class Program
    {
        static void Main()
        {
            int a = int.Parse(Console.ReadLine());
            for (int d = 1; d <=a; d++)
            { int b = d;
                if (b % 3 == 0) { Console.Write(" " + d);continue; }
                while (b!=0)
                {
                    if (b % 10 == 3) { Console.Write(" "+d); break; }
                    b /= 10;
                }
            }
            Console.WriteLine();
        }
    }
}
