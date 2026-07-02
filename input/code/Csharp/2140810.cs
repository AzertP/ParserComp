using System;

namespace _6_B
{
    class Program
    {
        static void Main(string[] args)
        {
            bool[] card = new bool[52];
            int n = int.Parse(Console.ReadLine());
            for (int i = 0; i < n; i++)
            {
                string[] x = Console.ReadLine().Split();
                int kigou = 0;
                if (x[0] == "H")
                {
                    kigou = 1;
                }
                if (x[0] == "C")
                {
                    kigou = 2;
                }
                if (x[0] == "D")
                {
                    kigou = 3;
                }
                card[kigou * 13 + int.Parse(x[1]) - 1] = true;
            }
            for(int i=0;i<4;i++)
            {
                for(int j=1;j<=13;j++)
                {
                    if(card[i*13+j-1]==false)
                    {
                        if (i == 0)
                        {
                            Console.WriteLine("S " + j);
                        }
                        if (i == 1)
                        {
                            Console.WriteLine("H " + j);
                        }
                        if (i == 2)
                        {
                            Console.WriteLine("C " + j);
                        }
                        if (i == 3)
                        {
                            Console.WriteLine("D " + j);
                        }
                    }
                }
            }
            Console.ReadLine();
        }
    }
}
