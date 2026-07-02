using System;

public class hello
{
    public static string Sharp(int n)
    {
        var result = "";
        if (n % 2 == 0)
        {
            for (int i = 0; i < n / 2; i++) result += "#.";
        }
        else
        {
            for (int i = 0; i < (n - 1) / 2; i++) result += "#.";
            result += "#";
        }
        return result;
    }
    public static string Dot(int n)
    {
        var result = "";
        if (n % 2 == 0)
        {
            for (int i = 0; i < n / 2; i++) result += ".#";
        }
        else
        {
            for (int i = 0; i < (n - 1) / 2; i++) result += ".#";
            result += ".";
        }
        return result;
    }
    public static void Main()
    {
        var flag = 1;
        while (flag == 1)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var h = int.Parse(line[0]);
            var w = int.Parse(line[1]);
            if (h == 0 && w == 0)
            {
                flag = 0;
                goto end;
            }
            var line1 = Sharp(w);
            var line2 = Dot(w);
            if (h % 2 == 0)
            {
                for (int i = 0; i < h / 2; i++)
                {
                    Console.WriteLine(line1);
                    Console.WriteLine(line2);
                }
                Console.WriteLine();
            }
            else
            {
                for (int i = 0; i < (h -1) / 2; i++)
                {
                    Console.WriteLine(line1);
                    Console.WriteLine(line2);
                }
                Console.WriteLine(line1);
                Console.WriteLine();
            }
            end:;
        }


    }
}
