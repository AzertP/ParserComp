using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApp9
{   
    class Program
    {
        public static void Main(string[] args) {
            long a = long.Parse(Console.ReadLine());
            long[] sss = Console.ReadLine().Split(' ').Select(long.Parse).ToArray();
            Array.Reverse(sss);
            for (int b = 0; b < a; b++)
            {   if(b+1==a)
                Console.Write("{0}", sss[b]);
                else
                Console.Write("{0} ",sss[b]);
                
            }
            Console.Write("\n");

        }
    }
    }
