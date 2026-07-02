using System;


public class hello
{
    public static bool IsPrime(int num)
    {
        if (num < 2) return false;
        else if (num == 2) return true;
        else if (num % 2 == 0) return false; // ??¶??°????????????????????????

        double sqrtNum = Math.Sqrt(num);
        for (int i = 3; i <= sqrtNum; i += 2)
        {
            if (num % i == 0)
            {
                return false;
            }
        }
        return true;
    }
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var count = 0;
        for (int i = 0; i < n; i++)
        {
            var p = int.Parse(Console.ReadLine().Trim());
            if (IsPrime(p)) count += 1;
        }
        Console.WriteLine(count);
    }
}
