using System.Collections.Generic;
using System;

public class hello
{
    public static void Main()
    {
        var arr = new List<int>();
        string[] line = Console.ReadLine().Trim().Split(' ');
        foreach(var n in line)   arr.Add(int.Parse(n));
        arr.Sort();
        Console.WriteLine("{0} {1} {2}", arr[0], arr[1], arr[2]);


    }
}
