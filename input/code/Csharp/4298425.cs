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

                var edge = new string('#', W);
                edge += "\n";

                var inner = new string('.', W - 2);
                var inner_line = "#" + inner + "#\n";
                var inner_rect = string.Concat(Enumerable.Repeat(inner_line, H - 2));

                var frame = edge + inner_rect + edge;

                Console.WriteLine(frame);
            }
        }
    }
}

