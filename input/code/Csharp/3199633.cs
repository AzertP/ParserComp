using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_6_D
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int[] line = ReadIntAr();
            int n = line[0];
            int m = line[1];

            int[,] matrixA = new int[n, m];
            int[] matrixB = new int[m];

            for(int i = 0; i < n; i++)
            {
                int[] lineA =ReadIntAr();
                for(int j = 0; j < m;j++)
                {
                    matrixA[i, j] = lineA[j];
                }
            }

            for (int i = 0; i < m; i++)
            {
                matrixB[i] = ReadInt();
            }

            long rlt;
            for (int i = 0; i < n; i++)
            {
                rlt = 0;
                for (int j = 0; j < m; j++)
                {
                    rlt += matrixA[i, j] * matrixB[j];
                }
                Console.WriteLine(rlt.ToString());
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

