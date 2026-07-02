using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_10_D
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n = ReadInt();
            int[] a = ReadIntAr();
            int[] b = ReadIntAr();
            int[] d = a.Select((x,index) => Math.Abs(x - b[index])).ToArray();

            double L1 = d.Sum();
            double L2 = Math.Sqrt(d.Sum(x => Math.Pow(x,2)));
            double L3 = Math.Pow(d.Sum(x => Math.Pow(x, 3)), 1/3D);
            double LINF = d.Max();

            Console.WriteLine("{0}", L1);
            Console.WriteLine("{0}", L2);
            Console.WriteLine("{0}", L3);
            Console.WriteLine("{0}", LINF);
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

