using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var s = Console.ReadLine();
            var p = Console.ReadLine();

            s += s;

            if (s.Contains(p))
            {
                Console.WriteLine("Yes");
            }
            else{
                Console.WriteLine("No");
            }
        }
    }
}

