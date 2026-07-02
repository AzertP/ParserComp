using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_5_A
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n = ReadInt();
            int[] target = ReadIntAr();
            int m = ReadInt();
            int[] searchKeys = ReadIntAr();

            for (int i = 0 ; i < m ; i++)
            {
                Console.WriteLine(IsCreateMAfterI(0, searchKeys[i], n, target) ? "yes":"no");
            }

        }

        public static bool IsCreateMAfterI(int i ,int m, int n, int[] a)
        {
            if (m == 0) return true;
            if (i >= n) return false;
            bool res = IsCreateMAfterI(i + 1, m, n, a) || IsCreateMAfterI(i + 1, m - a[i], n, a);
            return res;
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

