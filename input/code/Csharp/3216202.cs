using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_10_C
{
    public class Program

    {
        public static void Main(string[] args)
        {

            while (true)
            {
                int n = ReadInt();
                if (n == 0) return;

                int[] line = ReadIntAr();

                double ave = line.Average();
                double v = line.Sum(x => Math.Pow(x - ave, 2)) / n;
                Console.WriteLine("{0}",Math.Sqrt(v));
            }
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

