using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication17
{
    class Program
    {
        static void Main()
        {
            double[] a = Console.ReadLine().Split().Select(double.Parse).ToArray();
            double b = (a[2] - a[0]) * (a[2] - a[0]) + (a[3] - a[1]) * (a[3] - a[1]);
            Console.WriteLine(Math.Sqrt(b));
        }
    }
}
