using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ALDS1_3_D
{
    public class Program

    {
        public struct Pond
        {
            public int StartPoint { get;}
            public int Width { get;}
            public Pond(int startPoint, int width)
            {
                this.StartPoint = startPoint;
                this.Width = width;
            }
        }

        public static void Main(string[] args)
        {
            string line = ReadSt();
            Stack<Pond> ponds = new Stack<Pond>();
            Stack<int> areas = new Stack<int>();

            int startPoint = 0;
            int width = 0;

            for (int i = 0 ; i < line.Length ; i++)
            {
                if (line[i] == '\\')
                {
                    areas.Push(i);
                }
                else if (line[i] == '/' && areas.Count > 0)
                {
                    startPoint = areas.Pop();
                    width += i - startPoint;

                    while (ponds.Count > 0 && ponds.Peek().StartPoint > startPoint) //池の結合
                    {
                        width += ponds.Pop().Width;
                    }
                    ponds.Push(new Pond(startPoint, width));
                    width = 0;
                }
            }

            Console.WriteLine(ponds.Sum(x => x.Width));
            Console.WriteLine("{0}{1}", ponds.Count.ToString(), (ponds.Count > 0) ? " " + string.Join(" ", ponds.Select(x => x.Width.ToString()).Reverse().ToArray()): "");
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

