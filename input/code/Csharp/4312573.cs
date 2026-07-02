using System;

public class dice
{
    private int[] data;
    public dice(string s = "1,5,6,4,2,3")
    {
        data = new int[6];
        string[] line = s.Split(',');
        for (int i = 0; i < 6; i++)
            data[i] = int.Parse(line[i]);

        // 0: ceilling 1: front 2: floor 3:right:4:back:5:left
    }
    public void roll(char c)
    {
        var w = new int[6];
        Array.Copy(data, w, 6);
        if (c == 'U' | c == 'N') { data[4] = w[0]; data[0] = w[1]; data[1] = w[2]; data[2] = w[4]; }
        else if (c == 'D' | c == 'S') { data[1] = w[0]; data[2] = w[1]; data[4] = w[2]; data[0] = w[4]; }
        else if (c == 'R' | c == 'E') { data[3] = w[0]; data[5] = w[2]; data[2] = w[3]; data[0] = w[5]; }
        else { data[5] = w[0]; data[3] = w[2]; data[0] = w[3]; data[2] = w[5]; }
    }
    public int peek(int n) => data[n];
}

public class hello
{
    public static void Main()
    {
        var b0 = "013542";
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = Array.ConvertAll(line, int.Parse);
        var b = new int[6];
        for (int i = 0; i < 6; i++)
            b[b0[i] - '0'] = a[i];
        var d = new dice(string.Join(",", b));
        var s = Console.ReadLine().Trim();
        for (int i = 0; i < s.Length; i++)
            d.roll(s[i]);
        var ans = d.peek(0);
        Console.WriteLine(ans);
    }
}

