using System.Linq;
using System.Collections.Generic;
using System;

public class hello
{
    public static void Main()
    {
        var lst = new LinkedList<int>();

        var n = int.Parse(Console.ReadLine().Trim());
        for (int i = 0; i < n; i++)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var cmd = line[0];
            switch(cmd)
            {
                case "insert":
                    lst.AddFirst(int.Parse(line[1]));
                    break;
                case "delete":
                    lst.Remove(int.Parse(line[1]));
                    break;
                case "deleteFirst":
                    lst.RemoveFirst();
                    break;
                case "deleteLast":
                    lst.RemoveLast();
                    break;
            }
        }
        var count = 1;
        foreach (var x in lst)
        {
            Console.Write(x);
            if (count == lst.Count()) Console.WriteLine();
            else Console.Write(" ");
            count++;
        }
    }
}
