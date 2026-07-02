using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication1
{
    class Program
    {
        static void Main(string[] args)
        {
           while(true)
            {
                string[] sss = Console.ReadLine().Split();
                int a = int.Parse(sss[0]); int b = int.Parse(sss[1]); int c = int.Parse(sss[2]);int d = a + b;
                if (a == -1 && b == -1 && c == -1)
                    break;
                else if (d > 79)
                    Console.WriteLine("A");
                else if (a + b <= 79 && a + b > 64)
                    Console.WriteLine("B");
                else if (a + b > 49 && a + b <= 64)
                    Console.WriteLine("C");
                else if (a + b > 29 && a + b < 49 && c > 49)
                    Console.WriteLine("C");
                else if (a + b > 29 && a + b < 49 && c <= 49)
                    Console.WriteLine("D");
                else
                    Console.WriteLine("F");
            }
        }
    }
}

    
