using System.Collections.Generic;
using System;

public class Process
{
    public string pname { get; set; }
    public int ptime { get; set; }
}

public class hello
{
    public static void Main()
    {
        var que = new Queue<Process>();
        string[] line = Console.ReadLine().Trim().Split(' ');
        var n = int.Parse(line[0]);
        var q = int.Parse(line[1]);
        for (int i = 0; i < n; i++)
        {
            string[] line2 = Console.ReadLine().Trim().Split(' ');
            var a = new Process { pname = line2[0], ptime = int.Parse(line2[1] )};
            que.Enqueue(a);
        }
        var count = 0;
        while(que.Count > 0)
        {
            var s = que.Dequeue();
            if (s.ptime - q <=0 )
            {
                Console.WriteLine("{0} {1}", s.pname, count + s.ptime);
                count += s.ptime;
            }
            else
            {
                s.ptime -= q;
                que.Enqueue(s);
                count += q;
            }
        }


    }
}
