using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_7_D
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int[] line = ReadIntAr();
            int[,] matA = new int[line[0], line[1]];
            int[,] matB = new int[line[1], line[2]];
            long[,] rlt = new long[line[0], line[2]];

            for (int i = 0 ; i < line[0] ; i++)
            {
                int[] tmp = ReadIntAr();
                for (int j = 0 ; j < line[1] ; j++)
                {
                    matA[i, j] = tmp[j];
                }
            }

            for (int i = 0 ; i < line[1] ; i++)
            {
                int[] tmp = ReadIntAr();
                for (int j = 0 ; j < line[2] ; j++)
                {
                    matB[i, j] = tmp[j];
                }
            }

            for (int i = 0 ; i < line[0] ; i++)
            {
                for (int j = 0 ; j < line[2] ; j++)
                {
                    for (int k = 0 ; k < line[1] ; k++)
                    {
                        rlt[i, j] += matA[i, k] * matB[k, j];
                    }
                }
            }

            for (int i = 0 ; i < line[0] ; i++)
            {
                for (int j = 0 ; j < line[2] ; j++)
                {
                    Console.Write(rlt[i, j]);
                    if (j == line[2] - 1) Console.Write(Environment.NewLine);
                    else Console.Write(" ");
                }
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

