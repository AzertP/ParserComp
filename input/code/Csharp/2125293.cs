using System;
using System.Linq;
class Program {
	public static void Main() {
		int[] x = Console.ReadLine().Split().Select(int.Parse).ToArray();
		Console.WriteLine((x[0] / x[1]) + " " + (x[0] % x[1]) + " " + (1.0 * x[0] / x[1]).ToString("f12"));
	}
}
