using System;

namespace _9_B
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                string s = Console.ReadLine();
                if (s == "-")
                {
                    break;
                }
                int n = int.Parse(Console.ReadLine());
                for (int i = 0; i < n; i++)
                {
                    int x = int.Parse(Console.ReadLine());
                    s = s.Substring(x, s.Length - x) + s.Substring(0, x);
                }
                Console.WriteLine(s);
            }
            Console.ReadLine();
        }
    }
}
