using System;

namespace _8_C
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] ret = new int[500];
            while (true)
            {
                string s = Console.ReadLine();
                if (s == null) break;
                s = s.ToLower();
                for (int i = 0; i < s.Length; i++)
                {
                    ret[s[i]]++;
                }
            }
            for (int i = 'a'; i <= 'z'; i++)
            {
                Console.WriteLine((char)i + " : " + ret[i]);
            }
        }
    }
}
