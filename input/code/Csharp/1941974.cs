using System;
using System.Linq;

class P
{
  static void Swap<T>(ref T a, ref T b)
  {
    var c = a;
    a = b;
    b = c;
  }

  static void Main()
  {
    for(;;) {
      var a = Console.ReadLine().Split(' ').Select(x => int.Parse(x)).ToArray();
      if(a[0] > a[1])
        Swap(ref a[0], ref a[1]);
      if(a[1] == 0) break;
      Console.WriteLine("{0} {1}", a[0], a[1]);
    }
  }
}
