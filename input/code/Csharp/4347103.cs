using System;
using System.Linq;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                var deck = Console.ReadLine();
                if (deck == "-")
                {
                    break;
                }

                var m = int.Parse(Console.ReadLine());

                for (var i = 0; i < m; i++)
                {
                    var h = int.Parse(Console.ReadLine());
                    deck = deck.Substring(h) + deck.Substring(0, h);
                }

                Console.WriteLine(deck);
            }
        }
    }
}

