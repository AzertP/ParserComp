using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITP1_11_A
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

            public void TurnU()
            {
                int tmp = this.top;
                this.top = this.front;
                this.front = this.bottom;
                this.bottom = this.back;
                this.back = tmp;
            }

            public void TurnR()
            {
                int tmp = this.right;
                this.right = this.top;
                this.top = this.left;
                this.left = this.bottom;
                this.bottom = tmp;
            }
        }


        public static void Main(string[] args)
        {
            int[] line = ReadIntAr();
            Dice dice = new Dice(line);
            string cmd = ReadSt();
            for (int i = 0 ; i < cmd.Length ; i++)
            {
                switch (cmd[i])
                {
                    case 'N':
                        dice.TurnU();
                        break;
                    case 'S':
                        dice.TurnU(); dice.TurnU(); dice.TurnU();
                        break;
                    case 'E':
                        dice.TurnR();
                        break;
                    case 'W':
                        dice.TurnR(); dice.TurnR(); dice.TurnR();
                        break;
                }
            }
                 Console.WriteLine(dice.top);
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

