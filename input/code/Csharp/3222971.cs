using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_4_D
{
    public class Program

    {

        public class LoadCapacityComparer : IComparer<int>
        {
            private int[] items;
            private int k;

            public LoadCapacityComparer(int k, int[] items)
            {
                this.k = k;
                this.items = items;
            }

            public int Compare(int x, int y)
            {
                int sum =0;
                int TruckCount = 1;
                for (int i = 0 ; i < items.Count() ; i++)
                {
                    if (sum + items[i] <= x)
                    {
                        sum += items[i];
                    }
                    else
                    {
                        TruckCount++;
                        if (TruckCount > k) return -1;
                        sum = items[i];                      
                    }
                }
                return 1;   
            }
        }

        public static void Main(string[] args)
        {
            int[] line = ReadIntAr();
            int n = line[0];
            int k = line[1];

            int[] items = new int[n];
            for (int i = 0 ; i < n ; i++) items[i] = ReadInt();

            int itemMax = items.Max();
            int itemSum = items.Sum();

            LoadCapacityComparer lcc = new LoadCapacityComparer(k,items);
            int index = BinarySearch(itemMax,itemSum,lcc);

            Console.WriteLine(~index);

        }

        //●二分探索
        public static int BinarySearch(int left, int right, IComparer<int> comparer)
        {

            int lo = left;
            int hi = right;
            while (lo <= hi)
            {
                int i = GetMedian(lo, hi);
                int c = comparer.Compare(i,0);//yはダミーとして0固定
                if (c == 0) return i;
                if (c < 0) lo = i + 1;
                else hi = i - 1;
            }
            return ~lo;
        }
        private static int GetMedian(int low, int hi) { return low + ((hi - low) >> 1); }

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

