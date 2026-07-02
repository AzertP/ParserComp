using System;
using System.Linq;

namespace ITP1_11_A
{
	class Dice
	{
		public static int[] label = new int[6];
		public static void Roll(string s)
		{
			foreach (char c in s)
				switch (c)
				{
					case 'E': { East(); break; }
					case 'N': { North(); break; }
					case 'S': { South(); break; }
					case 'W': { West(); break; }
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
			Dice.Roll(Console.ReadLine());
			Console.WriteLine(Dice.label[0]);
		}
	}
}
