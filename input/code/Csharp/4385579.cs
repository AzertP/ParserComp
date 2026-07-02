using System;

public class ITP1_10_A{
    public static void Main(){
        var items = Console.ReadLine().Split(' ');
        
        var x1 = double.Parse(items[0]);
        var y1 = double.Parse(items[1]);
        var x2 = double.Parse(items[2]);
        var y2 = double.Parse(items[3]);
        
        var ans = Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));
        Console.WriteLine(ans);
    }
}

