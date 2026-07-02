using System;
namespace CSharp.consoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string s=Console.ReadLine();
            string[] numbers = s.Split(' ');
            int W=int.Parse(numbers[0]);
            int H=int.Parse(numbers[1]);
            int x=int.Parse(numbers[2]);
            int y=int.Parse(numbers[3]);
            int r=int.Parse(numbers[4]);
            if (x <r || y <r)
                Console.WriteLine("No");
            else
            {
                if (W >= x + r && H >= y + r)
                    Console.WriteLine("Yes");
                else
                    Console.WriteLine("No");
            }
            
        }
    }
}
