using System;

namespace _1_D
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = int.Parse(Console.ReadLine());
            int[] r = new int[n];
            for(int i=0;i<n;i++)
            {
                r[i] = int.Parse(Console.ReadLine());
            }
            int minv = r[0];
            int maxv = r[1] - r[0];
            for(int i=1;i<n;i++)
            {
                if(maxv<r[i]-minv)
                {
                    maxv = r[i] - minv;
                }
                if(minv>r[i])
                {
                    minv = r[i];
                }
            }
            Console.WriteLine(maxv);
            Console.ReadLine();
        }
    }
}
