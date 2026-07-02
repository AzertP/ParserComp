using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var W = Console.ReadLine();

            var cnt = 0;
            while (true)
            {
                var line = Console.ReadLine();
                if (line == "END_OF_TEXT")
                {
                    break;
                }

                var words = line.Split(' ').Select(x => x.ToLower());
                cnt += words.Where(x => x == W).Count();
            }

            Console.WriteLine(cnt);
        }
    }
}

