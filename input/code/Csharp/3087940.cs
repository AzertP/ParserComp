using System.Collections.Generic;
using System;

public class hello
{
    public static void Main()
    {
        var lst = new HashSet<string>();
        var n = int.Parse(Console.ReadLine().Trim());
        for (int i = 0; i < n; i++)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            if (line[0] == "insert") lst.Add(line[1]);
            else
            {
                if (lst.Contains(line[1])) Console.WriteLine("yes");
                else Console.WriteLine("no");
            }
        }
    }
}
