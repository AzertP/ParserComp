using System;

namespace _8_B
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                string now = Console.ReadLine();
                if (now == "0")
                {
                    break;
                }
                int output = 0;
                for (int i = 0; i < now.Length; i++)
                {
                    int x = now[i] - '0';
                    output += x;
                }
                Console.WriteLine(output);
            }
            Console.ReadLine();
        }
    }
}
