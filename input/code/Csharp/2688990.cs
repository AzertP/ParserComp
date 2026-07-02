using System;

public static class Program
{
	public static void Main()
	{
		int n = int.Parse(Console.ReadLine());
		int minValue = 1 << 30;
		int ans = -(1 << 30);
		for (int i = 0; i < n; i++)
		{
			int a = int.Parse(Console.ReadLine());
			if (i >= 1)
			{
				ans = Math.Max(ans, a - minValue);
			}
			minValue = Math.Min(a, minValue);
		}
		Console.WriteLine(ans);
	}
}

