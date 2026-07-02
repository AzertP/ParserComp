using System;


public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var defmax = -1000000000;
        var minvalue = int.Parse(Console.ReadLine().Trim());
        for (int i = 0; i < n - 1; i++)
        {
            var rvalue = int.Parse(Console.ReadLine().Trim());
            defmax = Math.Max(defmax, rvalue - minvalue);
            minvalue = Math.Min(minvalue, rvalue);
        }
        Console.WriteLine(defmax);
    }
}
