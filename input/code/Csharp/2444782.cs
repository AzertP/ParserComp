using System;


public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var house = new int[12, 10];
        for (int i = 0; i < n; i++)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var b = int.Parse(line[0]) - 1;
            var f = int.Parse(line[1]) - 1;
            var r = int.Parse(line[2]) - 1;
            var v = int.Parse(line[3]);
            house[f + b * 3, r] += v;
        }
        var igeta = "####################";
        var floorcount = 0;
        for (int i = 0; i < 12; i++)
        {
            if (floorcount ==3)
            {
                Console.WriteLine(igeta);
                floorcount = 0;
            }
            for (int j = 0; j < 10; j++)
            {
                if (j == 9) Console.WriteLine(" {0}", house[i, j]);
                else Console.Write(" {0}", house[i, j]);
            }
            floorcount += 1;
        }
    }
}
