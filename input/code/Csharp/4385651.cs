using System;

public class ITP1_10_B{
    public static void Main(){
        var items = Console.ReadLine().Split(' ');
        
        var a = double.Parse(items[0]);
        var b = double.Parse(items[1]);
        var C = double.Parse(items[2]);
        
        var C_rad = C / 180 * Math.PI;
        
        var S = a * b * Math.Sin(C_rad) / 2;
        
        var c = Math.Sqrt(a * a + b * b - 2 * a * b * Math.Cos(C_rad));
        var L = a + b + c;
        
        var h = 2 * S / a;
        
        Console.WriteLine(S);
        Console.WriteLine(L);
        Console.WriteLine(h);
    }
}

