using System;

public class hello
{
    public static void Main()
    {
        var seiseki = "";
        while (true)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var m = int.Parse(line[0]);
            var r = int.Parse(line[1]);
            var f = int.Parse(line[2]);
            if ((m==-1) && (r== -1) &&(f ==-1) )goto readend;

            if ( (m == -1) | (r == -1))
            {
                seiseki = "F";
                goto nextread;
            }
            if ( m+r >= 80)
            {
                seiseki = "A";
                goto nextread;
            }
            if   (  (m+r < 80) &&(m + r >= 65))
            {
                seiseki = "B";
                goto nextread;
            }
            if ((m + r < 65) && (m + r >= 50))
            {
                seiseki = "C";
                goto nextread;
            }
            if ((m + r < 50) && (m + r >= 30) &&(f >=50))
            {
                seiseki = "C";
                goto nextread;
            }
            if ((m + r < 50) && (m + r >= 30) && (f < 50))
            {
                seiseki = "D";
                goto nextread;
            }
            if ((m + r < 30) )
            {
                seiseki = "F";
                goto nextread;
            }
            nextread:;
            Console.WriteLine(seiseki);
       }
       readend:;
    }
}
