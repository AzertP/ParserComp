using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication26
{
    class Program
    {
        static void Main()
        { string s = Console.ReadLine();int i = 0;
            while (true)
            {
                string[] a = Console.ReadLine().Split();
                for(int b = 0; b < a.Length; b++)
                {
                    if (a[b] == "END_OF_TEXT") goto x;
                    if (s.Length != a[b].Length) continue;
                    for(int c = 0; c < a[b].Length; c++)
                    {
                        if (a[b][c] > 96)
                        {
                            if (a[b][c] != s[c] && a[b][c] != s[c] + 32) break;
                        }
                        else { if (a[b][c] != s[c] && a[b][c] != s[c] - 32) break; }
                        if (a[b].Length - 1 == c) i++;
                    }
                }
            }
        x:;Console.WriteLine(i);
        }
    }
}
