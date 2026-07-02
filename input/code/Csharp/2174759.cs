using System;
using System.Linq;

namespace _4_A
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = int.Parse(Console.ReadLine());
            int[] s = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int q = int.Parse(Console.ReadLine());
            int[] t = Console.ReadLine().Split().Select(int.Parse).ToArray();
            int x = 0;
            for (int i = 0; i < q; i++)
            {
                bool now = false;
                for (int j = 0; j < n; j++)
                {
                    if (s[j] == t[i])
                    {
                        now = true;
                    }
                }
                if (now)
                {
                    x++;
                }
            }
            Console.WriteLine(x);
        }
    }
}
