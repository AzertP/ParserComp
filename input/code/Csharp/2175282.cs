using System;

namespace _9_D
{
    class Program
    {
        static void Main(string[] args)
        {
            string s = Console.ReadLine();
            char[] str = new char[s.Length];
            for (int i = 0; i < s.Length; i++)
            {
                str[i] = s[i];
            }
            int n = int.Parse(Console.ReadLine());
            for (int i = 0; i < n; i++)
            {
                string[] x = Console.ReadLine().Split();
                if (x[0] == "print")
                {
                    for (int j = int.Parse(x[1]); j <= int.Parse(x[2]); j++)
                    {
                        Console.Write(str[j]);
                    }
                    Console.WriteLine();
                }
                else if (x[0] == "replace")
                {
                    int a = 0;
                    for (int j = int.Parse(x[1]); j <= int.Parse(x[2]); j++)
                    {
                        str[j] = x[3][a];
                        a++;
                    }
                }
                else
                {
                    char[] now = new char[int.Parse(x[2]) - int.Parse(x[1]) + 1];
                    int a = 0;
                    for (int j = int.Parse(x[1]); j <= int.Parse(x[2]); j++)
                    {
                        now[a] = str[j];
                        a++;
                    }
                    a = 0;
                    Array.Reverse(now);
                    for (int j = int.Parse(x[1]); j <= int.Parse(x[2]); j++)
                    {
                        str[j] = now[a];
                        a++;
                    }
                }
            }
        }
    }
}
