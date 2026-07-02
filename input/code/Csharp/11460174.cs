using System;

public class Program
{

    public static void Main()
    {
        long x = long.Parse( Console.ReadLine());

        long res = x*x*x;

        //long res1 = 1;
        //for (int i = 0; i < 3; i++)
        //{
        //    res1 *= x;
        //}

        Console.WriteLine(res);
    }
}
