using System.Collections.Generic;
using System;

public class hello
{
    public static void Main()
    {
        var alp = new List<string>() { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" };
        var text = "";
        string line;
        for (; (line = Console.ReadLine()) != null;)
        {
            text += line.Trim().ToLower();
        }
        var textlength = text.Length;
        foreach(var x in alp)
        {
            var buf = textlength - text.Replace(x, "").Length;
            Console.WriteLine("{0} : {1}",x, buf);
        }
    }
}
