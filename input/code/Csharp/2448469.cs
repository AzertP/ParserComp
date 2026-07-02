using System;

public class hello
{
    public static void Main()
    {
        var w = Console.ReadLine().Trim();
        var count = 0;
        while (true)
        {
            string[] s = Console.ReadLine().Trim().Split(' ');
            if (s[0] == "END_OF_TEXT") break;
            for (int i = 0; i < s.Length; i++)
                if (w.ToUpper() == s[i].ToUpper())
                    count++;
        }
        Console.WriteLine(count);

    }
}
