using System;
using System.Linq;

namespace _7_B
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                if (x[0] == 0 && x[1] == 0)
                {
                    break;
                }
                int a = 0;
                for (int i = 1; i <= x[0]; i++)
                {
                    for (int j = 1; j <= x[0]; j++)
                    {
                        for (int k = 1; k <= x[0]; k++)
                        {
                            if (i != j && j != k && k != i)
                            {
                                if (i + j + k == x[1])
                                {
                                    a++;
                                }
                            }
                        }
                    }
                }
                Console.WriteLine(a / 6);
            }
        }
    }
}
