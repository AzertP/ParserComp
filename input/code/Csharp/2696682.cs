using System;
using System.Collections.Generic;

public static class Program
{
	public class Process
	{
		public string name;
		public int time;
	}

	public static void Main()
	{
		var lineList = Console.ReadLine().Split(' ');
		int n = int.Parse(lineList[0]);
		int q = int.Parse(lineList[1]);
		var queue = new Queue<Process>();
		for (int i = 0; i < n; i++)
		{
			lineList = Console.ReadLine().Split(' ');
			var process = new Process()
			{
				name = lineList[0],
				time = int.Parse(lineList[1])
			};
			queue.Enqueue(process);
		}
		int time = 0;
		while (queue.Count > 0)
		{
			var process = queue.Dequeue();
			if (process.time <= q)
			{
				time += process.time;
				Console.WriteLine(string.Format("{0} {1}", process.name, time));
			}
			else
			{
				time += q;
				process.time -= q;
				queue.Enqueue(process);
			}
		}
	}
}

