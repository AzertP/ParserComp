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

            List<int> sss = new List<int>();
            var a = Console.ReadLine().Split().Select(int.Parse).ToArray();

            for (int b = 0; b < a[0]; b++)
            {
                var c= Console.ReadLine().Split().Select(int.Parse).ToArray();
                for(int d = 0; d < a[1]; d++)
                {
                    sss.Add(c[d]);
                }
            }

            for (int e = 0; e < a[0]; e++)
            {
                int g = 0;
                for(int f = 0; f < a[1]; f++)
                {
                    Console.Write(sss[e*a[1]+f]+" ");
                    g += sss[e * a[1] + f];
                }
                Console.WriteLine(g);
            }
            int i = 0;
            for(int h = 0; h < a[1]; h++)
            {
                int j = 0;
                for (int m = 0; m < a[0]*a[1]; m+=a[1])
                {
                    j += sss[m + h];
                }
                Console.Write(j+" ");
                i += j;
            }
            Console.WriteLine(i);

            }

        }
    }
