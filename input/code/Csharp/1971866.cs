using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication33
{
    class Program
    {
        static void Main()
        {
            string[] a = Console.ReadLine().Split();
            for(int b = 2; b < a.Length; b++)
            {    if (a[1] == null) break;
                if (!char.IsDigit(a[b][0])&&a[b].Length==1)
                {
                    switch (a[b][0]) {
                        case '+':a[b - 2] =( int.Parse(a[b - 2]) + int.Parse(a[b - 1])).ToString(); break;
                        case '*': a[b - 2] =( int.Parse(a[b - 2]) * int.Parse(a[b - 1])).ToString(); break;
                        case '-': a[b - 2] = (int.Parse(a[b - 2]) - int.Parse(a[b - 1])).ToString(); break;
                        default: a[b - 2] = (int.Parse(a[b - 2]) / int.Parse(a[b - 1])).ToString(); break;
                    }
                    for(int i = b - 1; i < a.Length; i++)
                    {
                        try { a[i] = a[i + 2]; }
                        catch { a[i] = null; }
                    }b = 1;
                }
                if (a[1] == null) break;
            }
            Console.WriteLine(a[0]);
        }
    }
}
