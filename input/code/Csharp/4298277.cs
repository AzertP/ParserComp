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

                var line = new string('#', W);
                line += "\n";

                var rect = string.Concat(Enumerable.Repeat(line, H));

                Console.WriteLine(rect);
            }
        }
    }
}

