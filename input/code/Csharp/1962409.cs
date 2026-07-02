using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication30
{
    class Program
    {
        static void Main()
        {
            List<int> R = new List<int>();
            int A = int.Parse(Console.ReadLine()), S = int.MinValue;int U = int.MinValue;
            for (int B = 0; B < A; B++)
            {
                int L = int.Parse(Console.ReadLine());
                R.Add(L);
            }
            for (int C = R.Count - 1; C > -1; C--)
            {
                    if (S >= R[C]||U>=R[C]) continue;
                    for (int D = C - 1; D > -1; D--)
                    {
                        if (R[C] - R[D] > S) { S = R[C] - R[D]; if (R[D] == 0) break;U = R[C]; }
                    }
            }
            Console.WriteLine(S);
            }
        }
    }
