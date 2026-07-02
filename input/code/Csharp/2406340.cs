using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using System.Linq.Expressions;
 
static class Program
{
    static void Main()
    {
        new Magatro().Solve();
    }
}
 
class Magatro
{
    private int N;
    private int[] A;
    private int cnt;
    private void Scan()
    {
        N = int.Parse(Console.ReadLine());
        A = new int[N];
        for (int i = 0; i < N; i++)
        {
            A[i] = int.Parse(Console.ReadLine());
        }
    }
    private void InsertionSort(int g)
    {
        for (int i = g; i < N; i++)
        {
            int v = A[i];
            int j = i - g;
            while (j >= 0 && A[j] > v)
            {
                A[j + g] = A[j];
                j = j - g;
                cnt++;
            }
            A[j + g] = v;
        }
    }
 
    public void Solve()
    {
        Scan();
        cnt = 0;
        List<int> g = new List<int>();
        g.Add(1);
        for(int i=1; ; i++)
        {
            int next = g[i - 1] * 3 + 1;
            if (next > N)
            {
                break;
            }
            g.Add(next);
        }
        g.Reverse();
        foreach(int i in g)
        {
            InsertionSort(i);
        }
   
        Console.WriteLine(g.Count);
        Console.WriteLine(string.Join(" ", g.Select(i => i.ToString()).ToArray()));
        Console.WriteLine(cnt);
        Console.WriteLine(string.Join("\n", A.Select(i => i.ToString()).ToArray()));
    }
}
