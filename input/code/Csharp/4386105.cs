using System;
using System.Linq;

public class ITP1_10_C{
    public static void Main(){
        while (true)
        {
            var n = int.Parse(Console.ReadLine());
            if (n == 0)
            {
                break;
            }
            
            var score = Console.ReadLine().Split().Select(double.Parse);
            
            var avg = score.Sum() / n;
            
            var variance = score.Select(x => Math.Pow(x - avg, 2)).Sum() / n;
            var sdev = Math.Sqrt(variance);
            
            Console.WriteLine(sdev);
        }
    }
}

