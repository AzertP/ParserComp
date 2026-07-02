using System;


public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = int.Parse(line[0]);
        var b = int.Parse(line[1]);
        var c = int.Parse(line[2]);
        var count = 0;
        for (int i = a; i < b + 1; i++)
            if (c % i == 0) count++;
        Console.WriteLine(count);
    }
}
