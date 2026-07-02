using System;
using System.Linq;

class P
{
  static void Main()
  {
    var a = Console.ReadLine().Split(' ').Select(x => int.Parse(x)).ToArray();
    Console.WriteLine("{0} {1} {2:f5}", a[0]/a[1], a[0]%a[1], Convert.ToDouble(a[0]) / a[1]);
  }
}
