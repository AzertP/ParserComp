using System;

namespace _5_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int N = int.Parse(Console.ReadLine());
            for (int i = 1; i <= N; i++)
            {
                if (i % 3 == 0 || i % 10 == 3 || i % 100 / 10 == 3 || i % 1000 / 100 == 3 || i % 10000 / 1000 == 3)
                {
                    Console.Write(" " + i);
                }
            }
            Console.WriteLine();
            Console.ReadLine();
        }
    }
}
