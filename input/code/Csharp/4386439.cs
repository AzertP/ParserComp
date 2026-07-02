using System;
using System.Linq;

public class ITP1_10_C{
    public static void Main(){
        var n = int.Parse(Console.ReadLine());
        var x = Console.ReadLine().Split().Select(double.Parse);
        var y = Console.ReadLine().Split().Select(double.Parse);
        
        var diff = x.Zip(y, (x_i, y_i) => Math.Abs(x_i - y_i));
        
        var man = diff.Sum();
        var euc = Math.Sqrt(diff.Select(v => v * v).Sum());
        var p3 = Math.Pow(diff.Select(v => Math.Pow(v, 3)).Sum(), 1.0 / 3);
        var chb = diff.Max();
        
        Console.WriteLine($"{man}\n{euc}\n{p3}\n{chb}");
    }
}

