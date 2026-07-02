using System;
using System.Linq;

namespace ALDS1_4_D
{
	class Program
	{
		static int limit;
		static int[] arr;

		static bool IsOK(int key)
		{
			int count = 0; int sum = 0;
			for (int i = 0; i < arr.Length; i++)
			{
				if (sum + arr[i] <= key) sum += arr[i];
				else
				{
					count++;
					sum = arr[i];
				}
			}
			count++;
			return limit >= count;
		}

		static int Binary_search()
		{
			int ng = -1;
			int ok = 1000000000;

			while (Math.Abs(ok - ng) > 1)
			{
				int mid = ng + (ok - ng) / 2;

				if (IsOK(mid)) ok = mid;
				else ng = mid;
			}

			return ok;
		}
		static void Main(string[] args)
		{
			int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
			arr = new int[x[0]]; limit = x[1];
			for (int i = 0; i < x[0]; i++) arr[i] = int.Parse(Console.ReadLine());
			Console.WriteLine(Math.Max(arr.Max(), Binary_search()));
		}
	}
}
