using System;

class P
{
  static void Main()
  {
    var r = double.Parse(Console.ReadLine());    
    Console.WriteLine("{0:f6} {1:f6}", Math.PI * r * r, 2 * Math.PI * r);
  }
}
