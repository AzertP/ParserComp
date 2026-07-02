using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication33
{
    class Program
    {
        static bool PrimeNumbers(int a)
        {
            if (a == 2) return true;
            for(int b=2; ; b++)
            {
                if (a % b == 0) return false;
                if ((b + 1) * (b + 1) > a) return true;
            }
        }


        static void Main()
        {
            int s = 0;
            int a = int.Parse(Console.ReadLine());
            for(int b = 0; b < a; b++)
            { int c = int.Parse(Console.ReadLine());
                if (PrimeNumbers(c)) s++;
            }
            Console.WriteLine(s);
        }
    }
}
