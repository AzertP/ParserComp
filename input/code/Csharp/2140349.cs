using System;

namespace _4_B
{
    class Program
    {
        static void Main(string[] args)
        {
            double r = double.Parse(Console.ReadLine());
            Console.WriteLine((r*r*Math.PI).ToString("f12") + " " + (r*2*Math.PI).ToString("f12"));
            Console.ReadLine();
        }
    }
}
