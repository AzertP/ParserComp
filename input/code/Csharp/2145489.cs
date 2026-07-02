using System;
using System.Linq;

namespace _6_C
{
    class Program
    {
        static void Main(string[] args)
        {
            int[,,] OfficialHouse = new int[4, 3, 10];
            int n = int.Parse(Console.ReadLine());
            for (int i = 0; i < n; i++)
            {
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                OfficialHouse[x[0] - 1, x[1] - 1, x[2] - 1] += x[3];
            }
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Console.Write(" " + OfficialHouse[0, i, j]);
                }
                Console.WriteLine();
            }
            for (int k = 1; k < 4; k++)
            {
                Console.WriteLine("####################");
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 10; j++)
                    {
                        Console.Write(" " + OfficialHouse[k, i, j]);
                    }
                    Console.WriteLine();
                }
            }
            Console.ReadLine();
        }
    }
}
