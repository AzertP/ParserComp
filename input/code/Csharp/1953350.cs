using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication14//abcdefghij
{
   class Program
    {
        static void Main()
        {
            string a = Console.ReadLine();
            char[] sss=new char [a.Length];
            for (int r = 0; r < a.Length; r++)
            {
                sss[r] = a[r];
            }
            int b = int.Parse(Console.ReadLine());
            for(int c = 0; c < b; c++)
            {
                string[] s = Console.ReadLine().Split();int f = int.Parse(s[1]), g = int.Parse(s[2]);
                switch (s[0][2]) {
                    case 'i':
                        for(int i = f; i < g+1; i++)
                        {
                            Console.Write(sss[i]);
                        }
                        Console.WriteLine();
                        break;
                    case 'v':
                            char[] sssss = new char[g-f+1];//5
                            for(int z = f; z < g+1; z++)
                            {
                                sssss[z-f] = sss[z];//0 3
                            }
                            for(int d = f; d < g+1; d++)
                            {
                                sss[d] = sssss[ g - d];
                            }
                        break;
                    case 'p':
                        for(int h = f; h < g+1; h++)
                        {
                            sss[h] = s[3][h-f];
                        }
                        break;
                }

            }
        }
    }
}
