using System;
using System.Linq;

class P
{
  static void Main()
  {
    var a = Console.ReadLine()
            .Split(' ')
            .Select(x => int.Parse(x))
            .ToArray();
    Array.Sort(a);
    Console.WriteLine("{0} {1} {2}", a[0], a[1], a[2]);
  }
}
