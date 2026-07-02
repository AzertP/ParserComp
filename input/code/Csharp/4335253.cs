using System;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            string line = Console.ReadLine();
            char[] chars;

            chars = line.ToCharArray(0, line.Length);

            foreach (var c in chars)
            {
                if (Char.IsLower(c))
                {
                    Console.Write(Char.ToUpper(c));
                }
                else
                {
                    Console.Write(Char.ToLower(c));
                }
            }

            Console.WriteLine();
        }
    }
}

