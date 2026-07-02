using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_10_B
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int[] line = ReadIntAr();
            double rad = line[2] * Math.PI / 180;
            double a = line[0];
            double b = line[1];

            double h = b * Math.Sin(rad);
            double c = Math.Sqrt(Math.Pow(a, 2) + Math.Pow(b, 2) - 2 * a * b * Math.Cos(rad));

            Console.WriteLine("{0}", a * h / 2);
            Console.WriteLine("{0}", a + b + c);
            Console.WriteLine(h);
        }

        static string ReadSt() { return Console.ReadLine(); }
        static int ReadInt() { return int.Parse(Console.ReadLine()); }
        static long ReadLong() { return long.Parse(Console.ReadLine()); }
        static double ReadDouble() { return double.Parse(Console.ReadLine()); }
        static string[] ReadStAr(char sep = ' ') { return Console.ReadLine().Split(sep); }
        static int[] ReadIntAr(char sep = ' ') { return Array.ConvertAll(Console.ReadLine().Split(sep), e => int.Parse(e)); }
        static long[] ReadLongAr(char sep = ' ') { return Array.ConvertAll(Console.ReadLine().Split(sep), e => long.Parse(e)); }
        static double[] ReadDoubleAr(char sep = ' ') { return Array.ConvertAll(Console.ReadLine().Split(sep), e => double.Parse(e)); }

    }

}

