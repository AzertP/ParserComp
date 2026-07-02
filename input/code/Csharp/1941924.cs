using System;

class P
{
  static void Main()
  {
    var a = Console.ReadLine().Split(' ');
    int W = int.Parse(a[0]), H = int.Parse(a[1]), x = int.Parse(a[2]), y = int.Parse(a[3]), r = int.Parse(a[4]);
    Console.WriteLine(x - r < 0 || x + r > W || y - r < 0 || y + r > H ? "No" : "Yes");
  }
}
