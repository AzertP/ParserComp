using System;

public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = int.Parse(line[0]);
        var b = int.Parse(line[1]);

        Console.WriteLine("{0} {1}",a*b,2*(a+b));

    }
}
