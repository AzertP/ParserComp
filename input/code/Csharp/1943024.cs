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
            int[,,] s = new int[4, 3, 10];
            int a = int.Parse(Console.ReadLine());
            for(int b = 0; b < a; b++)
            {
                int[] e = Console.ReadLine().Split().Select(int.Parse).ToArray();
                s[e[0]-1,e[1]-1,e[2]-1]+= e[3];
            }
            for(int c = 0; c < 4; c++)
            {
                for(int f = 0; f < 3; f++)
                {
                    for(int y = 0; y < 10; y++)
                    {
                       
                        Console.Write(" "+s[c,f,y]);
                    }
                    Console.WriteLine();
                }
                if (c != 3) Console.WriteLine("####################");
            }
        }
    }
}
