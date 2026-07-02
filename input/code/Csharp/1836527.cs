using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace ConsoleApp8
{   
    class Program
    {
        public static void Main(string[] args) {
            string[] input = Console.ReadLine().Split(' ');
            int a = int.Parse(input[0]); int b = int.Parse(input[1]); int c = int.Parse(input[2]); int f = 0;
            for (int d = a; d<= b; d++)
            { int e = c % d;    
                if (e == 0)
                    f++;
            }
            Console.WriteLine(f);
        }
    }
}
