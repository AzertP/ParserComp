using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var alp_num = new int[26];
            
            while (true)
            {
                string line = Console.ReadLine();
                if (line == null)
                {
                    break;
                }

                foreach (char c in line.ToCharArray())
                {
                    if (Char.IsLetter(c))
                    {
                        alp_num[Char.ToLower(c)-97]++;
                    }
                }
            }

            var i = 97;
            foreach (var n in alp_num)
            {
                Console.WriteLine($"{(Char)i++} : {n}");
            }
        }
    }
}

