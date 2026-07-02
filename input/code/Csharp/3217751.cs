using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_11_B
{
    public class Program

    {
        public class Dice
        {
            public int top;
            public int front;
            public int right;
            public int left;
            public int back;
            public int bottom;

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
            int[] line = ReadIntAr();
            int n = ReadInt();

            Dice dice = new Dice(line);
            String cmd = " ZZZXYYYZXXXZYYYZXXXYZZZ";

            for (int i = 0 ; i < n ; i++)
            {

                int[] topFront = ReadIntAr();

                for (int j = 0 ; j <= cmd.Length ; j++)
                {          
                    dice.Rotate(cmd[j]);
                    if (topFront[0] == dice.top && topFront[1] == dice.front)
                    {
                        Console.WriteLine(dice.right.ToString());
                        break;
                    }
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
