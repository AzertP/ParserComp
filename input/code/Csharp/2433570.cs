using System.Linq;
using System;

public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var arr = new long[n];
        string[] line = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < n; i++) arr[i] = int.Parse(line[i]);
        Console.WriteLine("{0} {1} {2}", arr.Min(), arr.Max(), arr.Sum());

    }
}
