using System.Linq;
using System.Collections.Generic;
using System;

public class P
{
    public int area { get; set; }
    public int id { get; set; }
}

public class Hello
{
    public static void Main()
    {
        var s = Console.ReadLine().Trim();
        getAns(s);
    }
    public static void print(Stack<P> st)
    {
        var n = st.Count();
        if (n == 0) { Console.WriteLine(0); Console.WriteLine(0); return; }
        var a = new int[n];
        var asum = 0;
        for (int i = 0; i < n; i++)
        {
            var w = st.Pop();
            a[n - 1 - i] = w.area;
            asum += w.area;
        }
        Console.WriteLine(asum);
        Console.Write("{0} ", n);
        for (int i = 0; i < n; i++)
            if (i == n - 1) Console.WriteLine(a[i]);
            else Console.Write("{0} ", a[i]);

    }
    public static void getAns(string s)
    {
        var st1 = new Stack<int>();
        var st2 = new Stack<P>();
        for (int i = 0; i < s.Length; i++)
            if (s[i] == '\\') st1.Push(i);
            else if (s[i] == '/' && st1.Count() > 0)
            {
                var id = st1.Pop();
                var area = i - id;
                while (st2.Count() > 0 && id <= st2.Peek().id)
                {
                    area += st2.Peek().area;
                    st2.Pop();
                }
                st2.Push(new P { id = id, area = area });
            }
        print(st2);
    }
}

