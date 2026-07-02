using System;
using System.Collections.Generic;
using System.Linq;

namespace _7_A
{
    class Program
    {
        static void Main(string[] args)
        {
            List<string> output = new List<string>();
            while (true)
            {
                int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
                int tyukan = x[0];
                int kimatu = x[1];
                int saisi = x[2];
                if(tyukan==-1&&kimatu==-1&&saisi==-1)
                {
                    break;
                }
                if (tyukan == -1 || kimatu == -1)
                {
                    output.Add("F");
                }
                else if (tyukan + kimatu >= 80)
                {
                    output.Add("A");
                }
                else if (tyukan + kimatu >= 65)
                {
                    output.Add("B");
                }
                else if (tyukan + kimatu >= 50)
                {
                    output.Add("C");
                }
                else if (tyukan + kimatu >= 30)
                {
                    if (saisi >= 50)
                    {
                        output.Add("C");
                    }
                    else
                    {
                        output.Add("D");
                    }
                }
                else
                {
                    output.Add("F");
                }
            }
            foreach(var i in output)
            {
                Console.WriteLine(i);
            }
        }
    }
}
