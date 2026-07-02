using System;

namespace _1_C
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = int.Parse(Console.ReadLine());
            int a = 0;
            for (int i = 0; i < n; i++)
            {
                int x = int.Parse(Console.ReadLine());
                bool z = true;
                if (x == 2)
                {
                    z = true;
                }
                else if (x < 2 || x % 2 == 0)
                {
                    z = false;
                }
                else
                {
                    int j = 3;
                    while (j <= Math.Sqrt(x))
                    {
                        if (x % j == 0)
                        {
                            z = false;
                        }
                        j = j + 2;
                    }
                }
                if (z == true)
                {
                    a++;
                }
            }
            Console.WriteLine(a);
            Console.ReadLine();
        }
    }
}
