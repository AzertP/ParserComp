using System;
using System.Collections.Generic;
using System.Linq;

namespace _3_A
{
    class Program
    {
        static void Main(string[] args)
        {
            Stack<int> number = new Stack<int>();
            string[] s = Console.ReadLine().Split();
            for (int i = 0; i < s.Count(); i++)
            {
                switch (s[i])
                {
                    case ("+"):
                        {
                            number.Push(number.Pop() + number.Pop());
                            break;
                        }
                    case ("-"):
                        {
                            int a = number.Pop();
                            number.Push(number.Pop() - a);
                            break;
                        }
                    case ("*"):
                        {
                            number.Push(number.Pop() * number.Pop());
                            break;
                        }
                    default:
                        {
                            number.Push(int.Parse(s[i]));
                            break;
                        }
                }
            }
            Console.WriteLine(number.Pop());
        }
    }
}
