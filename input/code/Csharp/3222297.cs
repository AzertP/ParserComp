using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_3_C
{
    public class Program

    {
        public static void Main(string[] args)
        {
            int n  = ReadInt();
            LinkedList<int> res = new LinkedList<int>();

            for (int i = 0 ; i < n ; i++)
            {
                string[] line = ReadStAr();

                switch (line[0])
                {
                    case "insert":
                        res.AddFirst(int.Parse(line[1]));
                        break;
                    case "delete":
                        res.Remove(int.Parse(line[1]));
                        break;
                    case "deleteFirst":
                        res.RemoveFirst();
                        break;
                    case "deleteLast":
                        res.RemoveLast();
                        break;
                }
            }

            Console.WriteLine(String.Join(" ",res.Select(x => x.ToString()).ToArray()));
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

