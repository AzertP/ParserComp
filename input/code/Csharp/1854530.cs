using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApp9
{   
    class Program
    {
        public static void Main(string[] args) {
            while (true)
            {
                string[] sss = Console.ReadLine().Split(' ');
                int a = int.Parse(sss[0]); int b = int.Parse(sss[1]);
                if (a==0&&b==0) { break; }
                for (int d =1;d<= a;d++)
                {
                    for (int c =1;c<=b;c++)
                    {
                        Console.Write("#");

                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
    }
}
