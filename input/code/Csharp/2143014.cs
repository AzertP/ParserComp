using System;

namespace _8_D
{
    class Program
    {
        static void Main(string[] args)
        {
            string s = Console.ReadLine();
            s = s + s;
            string p = Console.ReadLine();
            int n = s.IndexOf(p);
            if (n >= 0)
            {
                Console.WriteLine("Yes");
            }
            else
            {
                Console.WriteLine("No");
            }
            Console.ReadLine();
        }
    }
}
