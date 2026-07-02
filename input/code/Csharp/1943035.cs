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
            while (true)
            {
                string a = Console.ReadLine();
                if (a == "-") break;
                int s = 0;
                int c = int.Parse(Console.ReadLine());
                for(int d = 0; d < c; d++)
                {
                    int y = int.Parse(Console.ReadLine());
                    s += y;
                }
                int u = a.Length;
                int r = s % u;
                for(int t = 0; t < u - r; t++)
                {
                    Console.Write(a[r+t]);
                }
                for(int q = 0; q < r; q++)
                {
                    Console.Write(a[q]);
                }
                Console.WriteLine();
            }
        }
    }
}
