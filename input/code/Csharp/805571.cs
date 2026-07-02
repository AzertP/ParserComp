using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static void Main(string[] args) {
  	while(true) {
  	int n = int.Parse(Console.ReadLine());
  	if(n == 0) break;
  	var seq = Console.ReadLine().Split(' ').Select(x => double.Parse(x));
  	double avr = seq.Average();
  	double variance = seq.Select(x => (x - avr) * (x - avr)).Sum() / n;
  	Console.WriteLine(Math.Sqrt(variance));
  	}
  }
}
