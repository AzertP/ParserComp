using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_10_A
{
    public class Program

    {
        public static void Main(string[] args)
        {
            double[] line = ReadDoubleAr();
            Console.WriteLine("{0}",Math.Sqrt(Math.Pow((line[2]-line[0]),2)+Math.Pow((line[3] - line[1]), 2)));
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

