using System;
using System.Collections.Generic;
using System.Linq;

namespace _3_B
{
    class Program
    {
        static void Main(string[] args)
        {
            Queue<int> time = new Queue<int>();
            Queue<string> name = new Queue<string>();
            int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
            for (int i = 0; i < x[0]; i++)
            {
                string[] s = Console.ReadLine().Split();
                name.Enqueue(s[0]);
                time.Enqueue(int.Parse(s[1]));
            }
            int ret = 0;
            while (name.Count > 0)
            {
                int now = time.Dequeue();
                if (x[1] >= now)
                {
                    ret += now;
                    Console.WriteLine(name.Dequeue() + " " + ret);
                }
                else
                {
                    ret += x[1];
                    time.Enqueue(now - x[1]);
                    name.Enqueue(name.Dequeue());
                }
            }
        }
    }
}
