using System;

namespace _9_C
{
    class Program
    {
        static void Main(string[] args)
        {
            int kaisuu=int.Parse(Console.ReadLine());
            int taro = 0;
            int hanako = 0;
            for(int i=0;i<kaisuu;i++)
            {
                string[] x = Console.ReadLine().Split();
                string Taro = x[0];
                string Hanako = x[1];
                Array.Sort(x);
                if(Taro==Hanako)
                {
                    taro++;
                    hanako++;
                }
                else if(x[1]==Taro)
                {
                    taro += 3;
                }
                else
                {
                    hanako += 3;
                }
            }
            Console.WriteLine(taro + " " + hanako);
        }
    }
}
