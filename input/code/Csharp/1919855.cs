using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication3
{
    class Program
    {
        public static void Main()
        {
            while (true)
            {
                List<int> sss = new List<int>();
                var a = Console.ReadLine().Split().Select(int.Parse).ToArray();
                if (a[0] == 0 && a[1] == 0) break;

                for(int b = 1; b < a[0] - 1; b++)
                {
                    for(int c = b + 1; c < a[0]; c++)
                    {
                        if ((a[1] - b - c) == c||c> (a[1] - b - c)) break;
                        else if ((a[1] - b - c) - 1 == c) { sss.Add(b); sss.Add(c); sss.Add(a[1] - b - c); break; }
                        else if ((a[1] - b - c) > a[0]) continue;
                        else { sss.Add(b); sss.Add(c); sss.Add(a[1] - b - c); }
                    }
                }
               { Console.WriteLine(sss.Count/3); }

            }
        }
    }
}
