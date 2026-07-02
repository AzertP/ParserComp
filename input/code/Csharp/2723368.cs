using System;

namespace Submit
{
    class Program
    {
        static void Main(string[] args)
        {
            string a = Console.ReadLine();
            char[] b = a.ToCharArray();
            for (int i = 0; i < b.Length; i++)
            {
                if (b[i] <= 'z' && b[i] >= 'a') b[i] -= ' ';
                else if (b[i] <= 'Z' && b[i] >= 'A') b[i] += ' ';
            }
            Console.WriteLine(b);
        }
    }
}
