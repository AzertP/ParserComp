using System;//fAIR, LATER, OCCASIONALLY CLOUDY.
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication2
{
    class Program
    {
        static void Main(string[] args)
        {
            string a = Console.ReadLine();
            string b = "abcdefghijklmnopqrstuvwxyz";
            string c = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            int d = a.Length;
            int f;
            for (int e = 0; e < d; e++)
            {
               for( f =0;f!=26;f++)
                {
                   
                    if (a[e] == b[f]) { Console.Write(c[f]); break; }
                    else if (a[e] == c[f]) { Console.Write(b[f]); break; }
                   
                }
              if(f==26)   Console.Write(a[e]); 
            }
            Console.Write("\n");


        }
    }
}

//fAIR, LATER, OCCASIONALLY CLOUDY.
