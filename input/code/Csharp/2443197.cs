using System.Linq;
using System;


public class hello
{
    public static void Main()
    {
        var moji = Console.ReadLine().Trim();
        var n = int.Parse(Console.ReadLine().Trim());
        for (int i = 0; i < n; i++)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var a =int.Parse( line[1]);
            var b = int.Parse(line[2]);
            switch (line[0])
            {
                case "print":
                    var buf = moji.Substring(a , b - a + 1);
                    Console.WriteLine(buf);
                    break;
                case "reverse":
                    var buf1 = moji.Substring(0, a);
                    var buf2 = moji.Substring(a, b - a + 1);
                    var buf3 = moji.Substring(b +1);
                    buf1 += new string(buf2.Reverse().ToArray()) + buf3;
                    moji = buf1;
                    break;
                default:  //replace
                    var p = line[3];
                    var buf4 = moji.Substring(0, a);
                    var buf5 = moji.Substring(b +1);
                    buf4 += p + buf5;
                    moji = buf4;
                    break;
            }
        }
    }
}
