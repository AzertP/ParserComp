using System;


public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = int.Parse(line[0]);
        var b = int.Parse(line[1]);
        var c = int.Parse(line[2]);
        if (a < b && b < c) Console.WriteLine("Yes");
        else Console.WriteLine("No");
    }
}
