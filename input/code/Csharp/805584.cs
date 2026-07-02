using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static void Main(string[] args) {
  	double[] seq = Console.ReadLine().Split(' ').Select(x => double.Parse(x)).ToArray();
  	double a = seq[0];
  	double b = seq[1];
  	double rad = 2 * Math.PI * seq[2] / 360;
  	double S = a*b*Math.Sin(rad)/2;
  	double c = Math.Sqrt(a*a + b*b - 2*a*b*Math.Cos(rad));
  	Console.WriteLine(S);
  	Console.WriteLine(a+b+c);
  	Console.WriteLine(b*Math.Sin(rad));
  }
}
