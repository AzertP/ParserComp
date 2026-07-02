using System;
using System.Linq;

namespace _4_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int kosuu = Int32.Parse(Console.ReadLine());
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            long Sum=0;
            for(int i=0;i<kosuu;i++)
            {
                Sum += x[i];
            }
            Array.Sort(x);
            Console.WriteLine(x[0] + " " + x[kosuu-1] + " " + Sum);
        }
    }
}
