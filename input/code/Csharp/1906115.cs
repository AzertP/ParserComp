using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication1
{
    class Program
    {
        static StringBuilder sb = new StringBuilder();
        static int cnt;

        static void Main(string[] args)
        {
            int n = int.Parse(Console.ReadLine());
            int[] A = new int[n];

            for(int i= 0; i < n; i++)
            {
                A[i] = int.Parse(Console.ReadLine());
            }

            ShellSort(A, n);

            foreach (var num in A)
            {
                sb.AppendLine(num.ToString());
            }

            Console.Write(sb);
        }

        static void ShellSort(int[] A, int n)
        {
            cnt = 0;

            int h = 1, m = 0;

            do
            {
                m++;
                h = h * 3 + 1;

            } while (h < n);

            int[] G = new int[m];

            for(int i = m-1, x = 0; i >= 0; i--)
            {
                x = x * 3 + 1;
                G[i] = x;
            }
            
            for(int i = 0; i < m; i++)
            {
                InsertionSort(A, n, G[i]);
            }
            
            sb.AppendLine(m.ToString());
            sb.AppendLine(string.Join(" ", Array.ConvertAll(G, g => g.ToString())));
            sb.AppendLine(cnt.ToString());
        }

        static void InsertionSort(int[] A, int n, int g)
        {
            for(int i = g; i < n; i++)
            {
                int v = A[i];
                int j = i - g;

                while (j >= 0 && A[j] > v)
                {
                    A[j + g] = A[j];
                    j = j - g;
                    cnt++;
                }

                A[j + g] = v;
            }
        }
    }
}
