using System;
namespace ITP1_2_B {
	class Program {
		static void Main(string[] args) {
			string[] s = Console.ReadLine().Split();
			int[] a = new int[3];
			for (int i = 0; i < 3; i++) a[i] = int.Parse(s[i]);
			Array.Sort(a);
			Console.WriteLine(a[0] + " " + a[1] + " " + a[2]);
		}
	}
}
