using System;
using System.Linq;

namespace _3_C
{
    class Program
    {
        static void Main(string[] args)
        {
            bool z = true;
            while (z)
            {
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                if (x[0] == 0 && x[1] == 0)
                {
                    z = false;
                }
                else
                {
                    Array.Sort(x);
                    Console.WriteLine(x[0] + " " + x[1]);
                }
            }
        }
    }
}
