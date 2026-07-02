using System;

class P
{
  static void Main()
  {
    for(;;) {
      var s = Console.ReadLine().Split(' ');
      if(s[1] == "?") break;
      var a = int.Parse(s[0]);
      var b = int.Parse(s[2]);
      var r = 0;
      switch (s[1]) {
        case "+": r = a + b; break;
        case "-": r = a - b; break;
        case "*": r = a * b; break;
        case "/": r = a / b; break;
      }
      Console.WriteLine(r);
    }
  }
}
