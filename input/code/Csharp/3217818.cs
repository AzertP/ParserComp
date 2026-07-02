using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_1_A
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n = ReadInt();
            int[] line = ReadIntAr();
            InsertionSort(line);
        }

        public static void InsertionSort(int[] array)
        {
            Console.WriteLine(string.Join(" ",array.Select(x => x.ToString()).ToArray()));

            for (int i = 1 ; i < array.Length ; i++)
            {
                int v = array[i];
                int j = i - 1;
                while (j >= 0 && array[j] > v)
                {
                    array[j + 1] = array[j];
                    j--;
                }
                array[j + 1] = v;
                Console.WriteLine(string.Join(" ", array.Select(x => x.ToString()).ToArray()));
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

