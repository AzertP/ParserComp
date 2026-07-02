using System;
using System.Linq;

public static class Program
{
    public static void Main()
    {
        new Solve().Run();
    }
}

public class Solve
{
    int[] a;
    int[] query;
    int n;

    public void Run()
    {
        Console.ReadLine();
        a = Console.ReadLine().Split(' ').Select(x => int.Parse(x)).ToArray();
        n = a.Length;
        Console.ReadLine();
        query = Console.ReadLine().Split(' ').Select(x => int.Parse(x)).ToArray();
        foreach (var m in query)
        {
            Console.WriteLine(Func2(0, 0, m) ? "yes" : "no");
        }
    }

    public bool Func2(int i, int sum, int m)
    {
        if (i == n)
        {
            return sum == m;
        }
        return Func2(i + 1, sum, m) || Func2(i + 1, sum + a[i], m);
    }
}


