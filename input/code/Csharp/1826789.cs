using System;
namespace ITP1_2_B {
	class Program {
		static void Main(string[] args) {
			string[] s = Console.ReadLine().Split();
			int a = int.Parse(s[0]), b = int.Parse(s[1]), c = int.Parse(s[2]);
			Console.WriteLine((a < b && b < c) ? "Yes" : "No");
		}
	}
}
