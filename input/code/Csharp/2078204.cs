using System;
using System.Linq;
class Program {
	static void Main(String[] args) {
		int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
		Console.WriteLine((x[0] * x[1]) + " " + ((x[0] + x[1]) * 2));
	}
}
