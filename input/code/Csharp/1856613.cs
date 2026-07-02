using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApp9
{   
    class Program
    {
        public static void Main(string[] args) {
            long a = int.Parse(Console.ReadLine()), c = 0;
            long[] sss = Console.ReadLine().Split(' ').Select(long.Parse).ToArray();
            Array.Sort(sss);
            Console.Write("{0} {1} ", sss[0], sss[a - 1]);
            for (int b = 0; b < a; b++)
            {
                c = c + sss[b];

            }
            Console.WriteLine(c);

        }
    }
    }
