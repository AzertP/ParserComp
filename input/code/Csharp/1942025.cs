using System;
using System.Linq;

class P
{
  static void Main()
  {
    var a = Console.ReadLine().Split(' ').Select(x => int.Parse(x)).ToArray();
    var ans = 0;
    for(var i=a[0]; i<=a[1]; i++)
      ans += Convert.ToInt32(a[2] % i == 0);
    Console.WriteLine(ans);
  }
}
