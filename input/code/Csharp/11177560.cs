using System;

class Program
{
	static void Main()
	{
		int a, b;
		string[] input = Console.ReadLine().Split();
		a = int.Parse(input[0]);
		b = int.Parse(input[1]);

		if (-1000 <= a && b <= 1000)
		{
			if (a < b)
			{
				Console.WriteLine("a < b");
			}
			else if (a > b)
			{
				Console.WriteLine("a > b");
			}
			else
			{
				Console.WriteLine("a == b");
			}
		}
	}
}
