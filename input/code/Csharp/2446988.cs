using System;


public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = long.Parse(line[0]);
        var b = long.Parse(line[1]);
        var ans1 = a / b;
        var ans2 = a % b;
        var ans3 = (decimal)a / b;
        Console.WriteLine("{0} {1} {2}", ans1, ans2, ans3);

    }
}
