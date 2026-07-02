using System;

public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var atotal = 0;
        var btotal = 0;
        for (int i = 0; i < n; i++)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var a = line[0];
            var b = line[1];
            if (a.CompareTo(b) > 0) atotal += 3;
            else if (a.CompareTo(b) ==0)
            {
                atotal += 1;
                btotal += 1;
            }
            else btotal += 3;
        }
        Console.WriteLine("{0} {1}", atotal, btotal);
    }
}
