using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication15
{
    class Program
    {
        static void Main()
        {
            string a = Console.ReadLine();
            string b = Console.ReadLine();
            for(int c = 0; c < a.Length; c++)
            {
                for(int d = 0; d < b.Length; d++)
                {
                    if(c+d<a.Length){ if (a[c + d] != b[d]) break;
                        else if (b.Length - 1 == d) { Console.WriteLine("Yes"); goto y; } }
                    else
                    {
                        if (a[c + d-a.Length] != b[d]) break;
                        else if (b.Length - 1 == d) { Console.WriteLine("Yes"); goto y; }
                    }
                }
            }
            Console.WriteLine("No");    y:;
        }
    }
}
