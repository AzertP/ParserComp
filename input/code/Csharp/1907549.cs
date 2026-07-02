using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication1
{
    class Program
    {
        struct Pond_t
        {
            internal int pos;
            internal int area;
        }

        static void Main(string[] args)
        {
            string data = Console.ReadLine();

            Stack<int> downMap = new Stack<int>();
            Stack<Pond_t> ponds = new Stack<Pond_t>();

            int sumArea = 0;
            int len = data.Length;

            for (int i = 0; i < len; i++)
            {
                if (data[i] == '\\')
                {
                    downMap.Push(i);
                }
                else if (data[i] == '/' && downMap.Count > 0)
                {
                    int tempPos = downMap.Pop();
                    int tempArea = i - tempPos;
                    sumArea += tempArea;

                    while (ponds.Count > 0 && ponds.Peek().pos > tempPos)
                    {
                        tempArea += ponds.Pop().area;
                    }

                    Pond_t willPush;
                    willPush.pos = tempPos;
                    willPush.area = tempArea;
                    ponds.Push(willPush);
                }
            }

            Console.WriteLine(sumArea);
            Console.Write(ponds.Count);
            
            StringBuilder sb = new StringBuilder();
            var a = ponds.Select(p => p.area).ToArray();
            for(int i = a.Length - 1; i >= 0; i--)
            {
                sb.Append(" " + a[i]);
            }
            Console.WriteLine(sb);
        }
    }
}
