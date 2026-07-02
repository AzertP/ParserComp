using System;

public class hello
{
    public static void Main()
    {
        while (true)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var a= int.Parse(line[0]);
            var b = int.Parse(line[1]);
            if (a == 0 && b == 0) break;
            Console.WriteLine("{0} {1}",Math.Min(a,b), Math.Max(a, b));
        }
    }
}
