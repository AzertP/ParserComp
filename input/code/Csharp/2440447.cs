using System;


public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var ary = new int[n];
        string[]  line = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < n; i++) ary[i] = int.Parse(line[i]);
        var count = 0;
        var flag = 1;
        while ( flag == 1)
        {
            flag = 0;
            for (int i = n-1; i >=1; --i)
            {
                if (ary[i] < ary[i-1])
                {
                    count += 1;
                    var buf = ary[i - 1];
                    ary[i - 1] = ary[i];
                    ary[i] = buf;
                    flag = 1;
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (i == n - 1) Console.WriteLine(ary[i]);
            else Console.Write("{0} ", ary[i]);
        }
        Console.WriteLine(count);
    }
}
