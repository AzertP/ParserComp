using System;

class P
{
  static void Main()
  {
    for(var tc=1;; tc++) {
      var n = int.Parse(Console.ReadLine());
      if(0==n) break;
      Console.WriteLine("Case {0}: {1}", tc, n);
    }
  }
}
