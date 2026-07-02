using System;
using System.Linq;

namespace _9_A
{
    class Program
    {
        static void Main(string[] args)
        {
            string W = Console.ReadLine();
            W = W.ToLower();
            int a = 0;
            while (true)
            {
                string now = Console.ReadLine();
                if (now == "END_OF_TEXT")
                {
                    break;
                }
                now = now.ToLower();
                string[] x = now.Split();
                for (int i = 0; i < x.Count(); i++)
                {
                    if (x[i] == W)
                    {
                        a++;
                    }
                }
            }
            Console.WriteLine(a);
            Console.ReadLine();
        }
    }
}
