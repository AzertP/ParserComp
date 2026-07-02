using System;

namespace _4_C
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                string[] x = Console.ReadLine().Split();
                int a = Int32.Parse(x[0]);
                int b = Int32.Parse(x[2]);
                if (x[1] == "+")
                {
                    Console.WriteLine(a + b);
                }
                else if (x[1] == "-")
                {
                    Console.WriteLine(a - b);
                }
                else if (x[1] == "*")
                {
                    Console.WriteLine(a * b);
                }
                else if (x[1] == "/")
                {
                    Console.WriteLine(a / b);
                }
                else
                {
                    break;
                }
            }
        }
    }
}
