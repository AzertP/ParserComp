using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_11_C
{
    public class Program

    {
        public struct Dice
        {
            public int top { get; private set; }
            public int front { get; private set; }
            public int right { get; private set; }
            public int left { get; private set; }
            public int back { get; private set; }
            public int bottom { get; private set; }

            public Dice(int[] num)
            {
                this.top = num[0];
                this.front = num[1];
                this.right = num[2];
                this.left = num[3];
                this.back = num[4];
                this.bottom = num[5];
            }

            public void RotateX()//X軸に対し時計回りに回転
            {
                int tmp = this.top;
                this.top = this.back;
                this.back = this.bottom;
                this.bottom = this.front;
                this.front = tmp;
            }

            public void RotateY()//Y軸に対し時計回りに回転
            {
                int tmp = this.right;
                this.right = this.top;
                this.top = this.left;
                this.left = this.bottom;
                this.bottom = tmp;
            }

            public void RotateZ()//Z軸に対し時計回りに回転
            {
                int tmp = this.front;
                this.front = this.right;
                this.right = this.back;
                this.back = this.left;
                this.left = tmp;
            }

            public void Rotate(Char cmd)
            {
                switch (cmd)
                {
                    default: break;
                    case 'X': RotateX(); break;
                    case 'Y': RotateY(); break;
                    case 'Z': RotateZ(); break;
                }
            }

        }

        public static void Main(string[] args)
        {


            Dice diceA = new Dice(ReadIntAr());
            Dice diceB = new Dice(ReadIntAr());
            String cmd = " ZZZXYYYZXXXZYYYZXXXYZZZ";

            for (int j = 0 ; j < cmd.Length ; j++)
            {
                diceB.Rotate(cmd[j]);
                if (diceA.Equals(diceB))
                {
                    Console.WriteLine("Yes");
                    return;
                }
            }
            Console.WriteLine("No");

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



