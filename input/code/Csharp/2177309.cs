using System;

namespace _4_C
{
    class Program
    {
        static void Main(string[] args)
        {
            bool[] x = new bool[244140700];
            int n = int.Parse(Console.ReadLine());
            for (int i = 0; i < n; i++)
            {
                string[] s = Console.ReadLine().Split();
                int now = 0;
                int a = 1;
                for (int j = 0; j < s[1].Length; j++)
                {
                    switch (s[1][j])
                    {
                        case ('A'):
                            {
                                now += a;
                                break;
                            }
                        case ('C'):
                            {
                                now += a * 2;
                                break;
                            }
                        case ('G'):
                            {
                                now += a * 3;
                                break;
                            }
                        case ('T'):
                            {
                                now += a * 4;
                                break;
                            }
                    }
                    a *= 5;
                }
                if (s[0] == "insert")
                {
                    x[now] = true;
                }
                else
                {
                    Console.WriteLine(x[now] ? "yes" : "no");
                }
            }
            Console.ReadLine();
        }
    }
}
