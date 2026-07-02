using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication14
{
    class Program
    {
        static void Main()
        {
            List<string> sss = new List<string>();string s;
            int a = int.Parse(Console.ReadLine());
            for(int b = 0; b < a; b++)
            {
                string c = Console.ReadLine();
                sss.Add(c);
            }
            for(int d = 0; d < 4; d++)
            {
              
                for(int e = 1; e < 14; e++)
                {
                    switch (d) {
                        case 0:s = "S " + e; break;
                        case 1: s = "H " + e; break;
                        case 2: s = "C " + e; break;
                        default: s = "D " + e;break;
                    }

                    for(int f = 0; f < sss.Count; f++)
                    {
                        if (s == sss[f]) break;
                        else if (sss.Count - 1 == f) Console.WriteLine(s);
                    }
                }
            }
        }
    }
}
