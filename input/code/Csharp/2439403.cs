using System;

public class hello
{
    public static void Main()
    {
        while (true)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            if ((line[0] == "0") && (line[1]=="0"))goto readend;
            var h = int.Parse(line[0]);
            var w = int.Parse(line[1]);
            var s1 = "";
            var s2 = "";
            for (int i = 0; i < w; i++) s1 += "#";
            for (int i = 0; i < w-2; i++) s2 += ".";
            s2 = "#" + s2 + "#";
            Console.WriteLine(s1);
            for (int i = 0; i < h - 2;  i++)
                Console.WriteLine(s2);
            Console.WriteLine(s1);
            Console.WriteLine();
        }
        readend:;
    }
}
