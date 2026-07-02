using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                var items = Console.ReadLine().Split();
                
                var H = int.Parse(items[0]);
                var W = int.Parse(items[1]);

                if (H == 0 && W == 0)
                {
                    break;
                }

                var w_set = W / 2;
                var odd_line = string.Concat(Enumerable.Repeat("#.", w_set));
                var even_line = string.Concat(Enumerable.Repeat(".#", w_set));

                if (W % 2 == 1)
                {
                    odd_line += "#";
                    even_line += ".";
                }

                var lines = odd_line + "\n" + even_line + "\n";

                var h_set = H / 2;
                var board = String.Concat(Enumerable.Repeat(lines, h_set));

                if (H % 2 == 1)
                {
                    board += odd_line + "\n";
                }
                
                Console.WriteLine(board);
            }
        }
    }
}

