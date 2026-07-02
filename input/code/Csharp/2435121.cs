using System;

public class hello
{

    public static void Main()
    {
        var s = Console.ReadLine().Trim();
        var result = "";
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsUpper(s[i])) result += s[i].ToString().ToLower();
            else result += s[i].ToString().ToUpper();
        }
        Console.WriteLine(result);
    }
}
