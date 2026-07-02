using System;
using System.Linq;

namespace _3_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int a = x[0];
            int b = x[1];
            int c = x[2];
            int answer=0;
            for (int X=a;X<=b;X++)
            {
                if(c%X==0)
                {
                    answer++;
                }
            }
            Console.WriteLine(answer.ToString());
        }
    }
}
