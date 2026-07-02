using System;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                var line = Console.ReadLine().Split(' ');
                var n = int.Parse(line[0]);
                var x = int.Parse(line[1]);

                if (n == 0 && x == 0)
                {
                    break;
                }

                var lower1 = x / 3 + 1;
                var upper1 = (x - 3 < n) ? x - 3 : n;
                var cnt = 0;
                for (var n1 = lower1; n1 <= upper1; n1++)
                {
                    var rest = x - n1;
                    var lower2 = rest / 2 + 1;
                    var upper2 = (n1 - 1 < rest - 1) ? n1 - 1 : rest - 1;
                    cnt += upper2 - lower2 + 1;
                }

                Console.WriteLine(cnt);
            }
        }
    }
}

