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
                int a = int.Parse(sss[0]); int b = int.Parse(sss[2]);
                if (sss[1] =="?") { break; }
           else     if (sss[1] == "+") { Console.WriteLine(a+b); }
            else    if (sss[1] == "-") { Console.WriteLine(a - b); }
            else    if (sss[1] == "*") { Console.WriteLine(a * b); }
                else { Console.WriteLine(a / b); }

            }
        }
    }
}
