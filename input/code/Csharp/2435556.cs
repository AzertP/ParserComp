using System;

public class hello
{
    public static void Main()
    {
        while (true)
        {
            var moji = Console.ReadLine().Trim();
            if (moji == "0") goto end;
            var result = 0;
            for (int  i = 0; i < moji.Length; i++)
            {
                result += int.Parse(moji[i].ToString());
            }
            Console.WriteLine(result);
            result = 0;
       }
        end:;
    }
}
