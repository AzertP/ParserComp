using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_2_C
{
    public class Program

    {

        public struct Card
        {
                public char Suit { get; set; }
                public int Number { get; set; }

                public Card(String x)
            {
                Suit = x[0];
                Number = int.Parse(x[1].ToString());
            }

        }


        public static void Main(string[] args)
        {
            int n = ReadInt();
            string[] line = ReadStAr();
            Card[] Cards1 = line.Select(x => new Card(x)).ToArray();
            Card[] Cards2 = line.Select(x => new Card(x)).ToArray();

            BubbleSort(Cards1);
            SelectionSort(Cards2);

            Console.WriteLine(string.Join(" ",Cards1.Select(x => x.Suit.ToString() + x.Number.ToString()).ToArray()));
            Console.WriteLine("Stable");
            Console.WriteLine(string.Join(" ", Cards2.Select(x => x.Suit.ToString() + x.Number.ToString()).ToArray()));
            for (int i = 0 ; i < Cards1.Length ; i++)
            {
                if (Cards1[i].Suit != Cards2[i].Suit)
                {
                    Console.WriteLine("Not stable");
                    return;
                }
            }
            Console.WriteLine("Stable");
        }

        //●バブルソート
        public static int BubbleSort(Card[] array)
        {

            int swapTimes = 0;
            bool flg = true;
            while (flg)
            {
                flg = false;
                for (int i = array.Count() - 1 ; i > 0 ; i--)
                {
                    if (array[i].Number < array[i - 1].Number)
                    {
                        Card tmp = array[i];
                        array[i] = array[i - 1];
                        array[i - 1] = tmp;
                        swapTimes++;
                        flg = true;
                    }
                }
            }
            return swapTimes;
        }

        //●選択ソート
        public static int SelectionSort(Card[] array)
        {

            int swapTimes = 0;

            int min = 0;
            for (int i = 0 ; i < array.Count() ; i++)
            {
                min = i;
                for (int j = i ; j < array.Count() ; j++)
                {
                    if (array[j].Number < array[min].Number) min = j;
                }

                if (i != min)
                {
                    Card tmp = array[i];
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

