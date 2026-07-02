using System;
using System.Linq;

namespace ITP1_10_B
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] x = Console.ReadLine().Split().Select(double.Parse).ToArray();
            double s = x[0] * x[1] * Math.Sin(x[2] * (Math.PI / 180)) / 2;
            Console.WriteLine(s);
            double c = x[0] + x[1] + Math.Sqrt(Math.Pow(x[0], 2) + Math.Pow(x[1], 2) - (2 * x[0] * x[1] * Math.Cos(x[2] * (Math.PI / 180))));
            Console.WriteLine(c);
            double h = x[1] * Math.Sin(x[2] * (Math.PI / 180));
            Console.WriteLine(h);
        }
    }
}
