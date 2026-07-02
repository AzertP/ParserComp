using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.IO;
using System.Text;

class Program
{
    static StreamWriter sw = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = false };
    static Scan sc = new Scan();
    static void Main()
    {
        int n, q;
        sc.Multi(out n, out q);
        var prs = new proc[n];
        proc.q = q;
        for (int i = 0; i < n; i++)
        {
            prs[i] = new proc(i);
            sc.Multi(out prs[i].name, out prs[i].time);
        }
        Array.Sort(prs);
        long sum = 0;
        var bit = new BIT(n);
        for (int i = 0; i < n; ++i)
        {
            sw.WriteLine("{0} {1}", prs[i].name, prs[i].time + (prs[i].time - 1) / q * q * (n - i - 1) + (prs[i].ind - bit.sum(prs[i].ind)) * q + sum);
            sum += prs[i].time;
            bit.add(prs[i].ind, 1);
        }
        sw.Flush();
    }
    class proc : IComparable<proc>
    {
        public static int q;
        public string name;
        public long time;
        public int ind;
        public proc(int i)
        {
            ind = i;
        }
        public int CompareTo(proc p)
        {
            if ((time - 1) / q == (p.time - 1) / q)
                return ind.CompareTo(p.ind);

            return ((time - 1) / q).CompareTo((p.time - 1) / q);
        }
    }
}

class Scan
{
    public int Int { get { return int.Parse(Str); } }
    public long Long { get { return long.Parse(Str); } }
    public double Double { get { return double.Parse(Str); } }
    public string Str { get { return Console.ReadLine().Trim(); } }
    public int[] IntArr { get { return StrArr.Select(int.Parse).ToArray(); } }
    public long[] LongArr { get { return StrArr.Select(long.Parse).ToArray(); } }
    public double[] DoubleArr { get { return StrArr.Select(double.Parse).ToArray(); } }
    public string[] StrArr { get { return Str.Split(); } }
    bool eq<T, U>() { return typeof(T).Equals(typeof(U)); }
    T ct<T, U>(U a) { return (T)Convert.ChangeType(a, typeof(T)); }
    T cv<T>(string s) { return eq<T, int>()    ? ct<T, int>(int.Parse(s))
                             : eq<T, long>()   ? ct<T, long>(long.Parse(s))
                             : eq<T, double>() ? ct<T, double>(double.Parse(s))
                             : eq<T, char>()   ? ct<T, char>(s[0])
                                               : ct<T, string>(s); }
    public void Multi<T>(out T a) { a = cv<T>(Str); }
    public void Multi<T, U>(out T a, out U b)
    { var ar = StrArr; a = cv<T>(ar[0]); b = cv<U>(ar[1]); }
    public void Multi<T, U, V>(out T a, out U b, out V c)
    { var ar = StrArr; a = cv<T>(ar[0]); b = cv<U>(ar[1]); c = cv<V>(ar[2]); }
    public void Multi<T, U, V, W>(out T a, out U b, out V c, out W d)
    { var ar = StrArr; a = cv<T>(ar[0]); b = cv<U>(ar[1]); c = cv<V>(ar[2]); d = cv<W>(ar[3]); }
    public void Multi<T, U, V, W, X>(out T a, out U b, out V c, out W d, out X e)
    { var ar = StrArr; a = cv<T>(ar[0]); b = cv<U>(ar[1]); c = cv<V>(ar[2]); d = cv<W>(ar[3]); e = cv<X>(ar[4]); }
    public void Multi<T, U, V, W, X, Y>(out T a, out U b, out V c, out W d, out X e, out Y f)
    { var ar = StrArr; a = cv<T>(ar[0]); b = cv<U>(ar[1]); c = cv<V>(ar[2]); d = cv<W>(ar[3]); e = cv<X>(ar[4]); f = cv<Y>(ar[5]); }
    public void Multi<T, U, V, W, X, Y, Z>(out T a, out U b, out V c, out W d, out X e, out Y f, out Z g)
    { var ar = StrArr; a = cv<T>(ar[0]); b = cv<U>(ar[1]); c = cv<V>(ar[2]); d = cv<W>(ar[3]); e = cv<X>(ar[4]); f = cv<Y>(ar[5]);  g = cv<Z>(ar[6]);}
}

class BIT
{
    int n;
    long[] bit;
    public BIT(int n) { this.n = n; bit = new long[n]; }
    public void add(int j, long w) { for (int i = j; i < n; i |= i + 1) bit[i] += w; }
    // [0, j)
    public long sum(int j)
    {
        long ret = 0;
        for (int i = j - 1; i >= 0; i = (i & (i + 1)) - 1) ret += bit[i];
        return ret;
    }
    // [j, k)
    public long sum(int j, int k) { return sum(k) - sum(j); }
}
