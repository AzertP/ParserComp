using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AthleteProgramming
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = getint();
            string[] input = getsplit();
            string[] a = new string[n];
            string[] b = new string[n];
            for (int i = 0; i < n; i++)
            {
                a[i] = input[i];
                b[i] = input[i];

            }
            for (int i = 0; i < n; i++)
            {
                for (int j = n - 1; j > i; j--)
                {
                    if (a[j][1] < a[j - 1][1])
                    {
                        string k = a[j];
                        a[j] = a[j - 1];
                        a[j - 1] = k;
                    }
                }
            }
            for (int i = 0; i < n - 1; i++)
            {
                Console.Write(a[i] + " ");
            }
            Console.WriteLine(a[n - 1]);
            Console.WriteLine("Stable");
            for (int i = 0; i < n; i++)
            {
                int minj = i;
                for (int j = i; j < n; j++)
                {
                    if (b[j][1] < b[minj][1])
                    {
                        minj = j;
                    }
                }
                if (i != minj)
                {
                    string k = b[i];
                    b[i] = b[minj];
                    b[minj] = k;
                }
            }
            for (int i = 0; i < n - 1; i++)
            {
                Console.Write(b[i] + " ");
            }
            Console.WriteLine(b[n - 1]);
            bool s = true;
            for (int i = 0; i < n; i++)
            {
                if (a[i] != b[i])
                {
                    s = false;
                    break;
                }
            }



            if (s)
            {
                Console.WriteLine("Stable");
            }
            else
            {
                Console.WriteLine("Not stable");
            }
        }
        static string[] getsplit()
        {
            string[] x = Console.ReadLine().Split(' ');
            return x;
        }
        static int getint()
        {
            int x = int.Parse(Console.ReadLine());
            return x;
        }
    }
}
