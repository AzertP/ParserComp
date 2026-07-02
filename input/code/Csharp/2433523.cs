using System;

public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = int.Parse(line[0]);
        var b = int.Parse(line[1]);
        if (a > b) Console.WriteLine("a > b");
        else   if (a == b) Console.WriteLine("a == b");
        else Console.WriteLine("a < b");


    }
}
