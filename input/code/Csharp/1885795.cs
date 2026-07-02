using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication2
{
    class Program
    {
        static void Main(string[] args)
        {
            string a;
            while (true)
            {
                 a = Console.ReadLine();
                if (a == "0")break;
                int d = a.Length;int e = 0;
                for(int c = 0; c < d; c++)
                {
                    int k = (int)char.GetNumericValue(a[c]);
                    e = e + k;
                }if (a == "0") break;
                Console.Write(e+"\n");
            }
          
        }
    }
}
