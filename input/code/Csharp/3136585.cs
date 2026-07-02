using System;
using System.Linq;

namespace ITP1_10_D
{
	class Program
	{
		static void Main(string[] args)
		{
			int n = int.Parse(Console.ReadLine());
			int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
			int[] y = Console.ReadLine().Split().Select(int.Parse).ToArray();
			for (double z = 1; z <= 3; z++)
			{
				double sum = 0;
				for (int i = 0; i < n; i++)
				{
					sum += Math.Pow(Math.Abs(x[i] - y[i]), z);
				}
				Console.WriteLine(Math.Pow(sum, 1 / z));
			}
			double max = 0;
			for (int i = 0; i < n; i++) max = Math.Max(max, Math.Abs(x[i] - y[i]));
			Console.WriteLine(max);
		}
	}
}
