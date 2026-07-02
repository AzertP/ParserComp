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
            int[] c = new int[26];
            while (true)
            {    string a = Console.ReadLine();
                if (a == null) break;
                for (int b = 0; b < a.Length; b++)
                {
                    for (int d = 65; d < 91; d++)
                    {
                        if (a[b] == (char)(d + 32) || a[b] == (char)d) { c[d - 65]++; break; }
                    }
                }
            }

            for(int e = 0; e < 26; e++)
            {
                Console.WriteLine("{0} : {1}",(char)(e+97),c[e]);
            }
        }
    }
}
