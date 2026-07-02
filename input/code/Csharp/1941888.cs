using System;
using System.Linq;

class P
{
  static void Main()
  {
    var S = Console.ReadLine().Split(' ');
    var a = int.Parse(S[0]);
    var b = int.Parse(S[1]);
    if(a < b)
      Console.WriteLine("a < b");
    else if(a > b)
      Console.WriteLine("a > b");
    else
      Console.WriteLine("a == b");
  }
}
