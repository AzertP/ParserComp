using System;
using System.Linq;

namespace _2_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] X = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int W = X[0];
            int H = X[1];
            int x = X[2];
            int y = X[3];
            int r = X[4];
            if (x >= r && x + r <= W && y >= r && y + r <= H)
            {
                Console.WriteLine("Yes");
            }
            else
            {
                Console.WriteLine("No");
            }
        }
    }
}
