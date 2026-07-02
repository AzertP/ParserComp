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
                string line = Console.ReadLine();

                if (line == "0")
                {
                    break;
                }

                var digits = line.ToCharArray().Select(x => int.Parse(x.ToString()));
                var ans = digits.Sum();

                Console.WriteLine(ans);
            }
        }
    }
}

