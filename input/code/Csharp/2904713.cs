using System;

public class Hello
{
    public static void Main()
    {
        while (true)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var h = int.Parse(line[0]);
            var w = int.Parse(line[1]);
            if (h == 0 && w == 0) break;
            var s = new string('#', w);
            for (int i = 0; i < h; i++) Console.WriteLine(s);
            Console.WriteLine();
        }
    }
}
