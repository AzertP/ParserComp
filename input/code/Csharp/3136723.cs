using System;
using System.Linq;

namespace ITP1_11_B
{
	class Dice
	{
		public static int[] label = new int[6];
		public static void Roll(string s)
		{
			foreach (char c in s)
			{
				switch (c)
				{
					case 'E': { East(); break; }
					case 'N': { North(); break; }
					case 'S': { South(); break; }
					case 'W': { West(); break; }
				}
			}
		}
		static void East()
		{
			int x = label[0];
			label[0] = label[3]; label[3] = label[5];
			label[5] = label[2]; label[2] = x;
		}
		static void North()
		{
			int x = label[0];
			label[0] = label[1]; label[1] = label[5];
			label[5] = label[4]; label[4] = x;
		}
		static void South()
		{
			int x = label[0];
			label[0] = label[4]; label[4] = label[5];
			label[5] = label[1]; label[1] = x;
		}
		static void West()
		{
			int x = label[0];
			label[0] = label[2]; label[2] = label[5];
			label[5] = label[3]; label[3] = x;
		}
	}
	class Program
	{
		static void Main(string[] args)
		{
			Dice.label = Console.ReadLine().Split().Select(int.Parse).ToArray();
			string[] c = new string[] { "E", "N", "S", "W", };
			int n = int.Parse(Console.ReadLine());
			for (int i = 0; i < n; i++)
			{
				int[] a = Console.ReadLine().Split().Select(int.Parse).ToArray();
				Random rand = new Random();
				while (!(Dice.label[0] == a[0] && Dice.label[1] == a[1]))
				{
					int b = (int)(rand.NextDouble() * 100000);
					Dice.Roll(c[b % 4]);
					rand = new Random(b);
				}
				Console.WriteLine(Dice.label[2]);
			}
		}
	}
}
