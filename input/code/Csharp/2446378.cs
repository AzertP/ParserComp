using System;


public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var w = int.Parse(line[0]);
        
        var h = int.Parse(line[1]);
        var x = int.Parse(line[2]);
        var y = int.Parse(line[3]);
        var r = int.Parse(line[4]);
        if ((r <= x && x <= w - r) && (r <= y & y <= h - r)) Console.WriteLine("Yes");
        else Console.WriteLine("No");


    }
}
