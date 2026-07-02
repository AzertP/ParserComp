using System;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var nml = Console.ReadLine().Split(' ');
            var n = int.Parse(nml[0]);
            var m = int.Parse(nml[1]);
            var l = int.Parse(nml[2]);

            var A = new int[n, m];
            for (var i = 0; i < n; i++)
            {
                var items = Console.ReadLine().Split();
                for (var j = 0; j < m; j++)
                {
                    A[i, j] = int.Parse(items[j]);
                }
            }

            var B = new int[m, l];
            for (var i = 0; i < m; i++)
            {
                var items = Console.ReadLine().Split();
                for (var j = 0; j < l; j++)
                {
                    B[i, j] = int.Parse(items[j]);
                }
            }

            var C_i = new long[l];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < l; j++)
                {
                    long c_ij = 0;
                    for (var k = 0; k < m; k++)
                    {
                        c_ij += A[i, k] * B[k, j];
                    }
                    C_i[j] = c_ij;
                }
                Console.WriteLine(String.Join(" ", C_i));
            }
        }
    }
}

