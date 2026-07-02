using System;
public class Program
{
    public static void Main()
    {
        string input = Console.ReadLine();

        int spaceIndex = input.IndexOf(' ');

        int x = int.Parse(input.Substring(0, spaceIndex));
        int y = int.Parse(input.Substring(spaceIndex + 1));


        int a = x * y;
        int b = 2*(x + y);

        Console.WriteLine("{0} {1}", a,b);
    }
}
