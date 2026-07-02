using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_2_A
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n = ReadInt();
            int[] line = ReadIntAr();

            int swapTimes = BubbleSort(line);

            Console.WriteLine(string.Join(" ",line.Select(x => x.ToString()).ToArray()));
            Console.WriteLine(swapTimes);
        }

        //●バブルソート
        public static int BubbleSort(int[] array)
        {

            int swapTimes = 0;
            bool flg = true;
            while (flg)
            {
                flg = false;
                for (int i = array.Count() - 1 ; i > 0 ; i--)
                {
                    if (array[i] < array[i-1])
                    {
                        int tmp = array[i];
                        array[i] = array[i - 1];
                        array[i - 1] = tmp;
                        swapTimes++;
                        flg = true;
                    }
                }
            }
            return swapTimes;
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

