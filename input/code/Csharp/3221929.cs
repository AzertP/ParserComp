using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_2_B
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n = ReadInt();
            int[] line = ReadIntAr();

            int swapTimes = SelectionSort(line);

            Console.WriteLine(string.Join(" ", line.Select(x => x.ToString()).ToArray()));
            Console.WriteLine(swapTimes);
        }

        //●選択ソート
        public static int SelectionSort(int[] array)
        {

            int swapTimes = 0;

            int min = 0;
            for (int i = 0 ; i < array.Count() ; i++)
            {
                min = i;
                for (int j = i ; j < array.Count() ; j++)
                {
                    if(array[j] < array[min]) min = j;
                }

                if (i != min)
                {
                    int tmp = array[i];
                    array[i] = array[min];
                    array[min] = tmp;
                    swapTimes++;
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

