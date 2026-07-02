using System;
using System.Linq;

namespace _10_A
{
    class Program
    {
        static void Main()
        {
            double[] x = Console.ReadLine().Split().Select(double.Parse).ToArray();
            Console.WriteLine(Math.Sqrt((x[0] - x[2]) * (x[0] - x[2]) + (x[1] - x[3]) * (x[1] - x[3])));
            Console.ReadLine();
        }
    }
}
