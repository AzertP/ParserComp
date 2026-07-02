using System;
using System.Collections.Generic;

namespace ConsoleApplication1
{
    class Program
    {
        static void Main()
        {
            List<int> a = new List<int>();
            string str;
            str = Console.ReadLine();
            for (int i = 0;i < 3 ; i++)
            {
                a.Add(int.Parse(str.Split(' ')[i]));
            }
            a.Sort();
            Console.WriteLine(a[0] + " " + a[1] + " " + a[2] );
        }
    }
}
