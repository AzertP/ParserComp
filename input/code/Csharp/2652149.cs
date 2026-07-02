using System;
using System.Linq;

namespace ALDS1_4_B
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.ReadLine();
            int[] x1 = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int n2 = int.Parse(Console.ReadLine());
            int[] x2 = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int ret = 0;
            for (int i = 0; i < n2; i++)
            {
                if (Array.BinarySearch(x1,x2[i]) >= 0) ret++;
            }
            Console.WriteLine(ret);
        }
    }
}
