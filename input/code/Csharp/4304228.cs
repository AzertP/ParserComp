using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var line = Console.ReadLine().Split();
            var n = int.Parse(line[0]);
            var m = int.Parse(line[1]);

            var A = new int[n, m];
            
            for (var i = 0; i < n; i++)
            {
                line = Console.ReadLine().Split();
                for (var j = 0; j < m; j++)
                {
                    A[i, j] = int.Parse(line[j]);
                }
            }

            for (var i = 0; i < m; i++)
            {
                var b = int.Parse(Console.ReadLine());
                for (var j = 0; j < n; j++)
                {
                    A[j, i] *= b;
                }
            }

            for (var i = 0; i < n; i++)
            {
                var s = 0;
                for (var j = 0; j < m; j++)
                {
                    s += A[i, j];
                }
                Console.WriteLine(s);
            }
        }
    }
}

