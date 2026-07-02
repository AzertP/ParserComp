using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_2_D
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n = ReadInt();
            int[] line = new int[n];            
            for (int i = 0 ; i < n ; i++) line[i] = ReadInt();

            List<object> res = ShellSort(line);
            int swapTimes = (int)res[0];
            List<int> interval = (List<int>)res[1];

            Console.WriteLine(interval.Count);
            Console.WriteLine(string.Join(" ",interval.OrderByDescending(x => x).Select(x => x.ToString()).ToArray()));
            Console.WriteLine(swapTimes);
            for (int i = 0 ; i < n ; i++) Console.WriteLine(line[i].ToString());
        }

        //●シェルソート
        public static List<object> ShellSort(int[] array)
        {
            int swapTimes = 0;
            List<int> interval = new List<int>();
            for (int i = 1 ; i <= array.Length ; i = i * 3 + 1) interval.Add(i);

            for (int i = interval.Count - 1 ; i >= 0 ; i--)
            {
                swapTimes += InsertionSort(array, interval[i]);
            }

            List<object> res = new List<object>();
            res.Add(swapTimes);
            res.Add(interval);
            return res;
        }

        //●挿入ソート
        public static int InsertionSort(int[] array, int interval)
        {
            int swapTimes = 0;
            for (int i = interval ; i < array.Length ; i++)
            {
                int v = array[i];
                int j = i - interval;
                while (j >= 0 && array[j] > v)
                {
                    array[j + interval] = array[j];
                    j -= interval;
                    swapTimes++;
                }
                array[j + interval] = v;
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

