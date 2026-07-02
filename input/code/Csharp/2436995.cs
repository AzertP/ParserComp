using System.Linq;
using System;
public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var r = int.Parse(line[0]);
        var c = int.Parse(line[1]);
        var ary = new int[r, c + 1];
        var sumtate = new int[c + 1];    //??????????¨???????????????¨???
        for (int i = 0; i < r; i++)
        {
            string[] line2 = Console.ReadLine().Trim().Split(' ');
            var sumyoko = 0;
            for (int j = 0; j < c; j++)
            {
                ary[i, j] = int.Parse(line2[j]);
                sumyoko += ary[i, j];
            }
            ary[i, c] = sumyoko;
        }
        for (int i = 0; i < c; i++)
            for (int j = 0; j < r; j++)
                sumtate[i] += ary[j, i];
        var buf = sumtate.Sum();
        sumtate[c] = buf;

        for (int i = 0; i< r; i++)
            for (int j = 0; j< c+1; j++)
            {
                if (j != c) Console.Write("{0} ", ary[i, j]);
                else Console.WriteLine(ary[i, j]);
            }
        for (int i = 0; i< c+1; i++)
        {
            if (i != c) Console.Write("{0} ", sumtate[i]);
            else Console.WriteLine(sumtate[i]);

        }
    }
}
