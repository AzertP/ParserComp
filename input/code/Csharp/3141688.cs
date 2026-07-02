using System;
using System.Collections.Generic;

namespace ALDS1_3_C
{
	class Program
	{
		static void Main(string[] args)
		{
			LinkedList<string> list = new LinkedList<string>();
			int n = int.Parse(Console.ReadLine());
			for (int i = 0; i < n; i++)
			{
				string[] s = Console.ReadLine().Split();
				switch (s[0])
				{
					case "insert":
					{
						list.AddFirst(s[1]);
						break;
					}
					case "delete":
					{
						//if (list.Contains(s[1])) 
						list.Remove(s[1]);
						break;
					}
					case "deleteFirst":
					{
						list.RemoveFirst();
						break;
					}
					case "deleteLast":
					{
						list.RemoveLast();
						break;
					}
				}
			}
			Console.WriteLine(string.Join(" ", list));
		}
	}
}
