using System.Collections.Generic;
using System;

public class hello
{
    public static void Main()
    {
        var s = new Stack<int>();
        var acu = 0;
        string[] line = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < line.Length; i++)
        {
            switch(line[i])
            {
                case "+":
                    acu = s.Pop();
                    acu += s.Pop();
                    s.Push(acu);
                    break;
                case "-":
                    acu = s.Pop();
                    acu -= s.Pop();
                    s.Push(-acu);
                    break;
                case "*":
                    acu = s.Pop();
                    acu *= s.Pop();
                    s.Push(acu);
                    break;
                default:
                    s.Push(int.Parse(line[i]));
                    break;
            }
        }
        Console.WriteLine(s.Pop());
    }
}
