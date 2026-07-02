using System;
using System.Linq;
using System.Collections.Generic;

class Program {
  static void Main() {
    var a = new LinkedList<string>();
    int n = int.Parse(Console.ReadLine());
    for (int i = 0; i < n; i++) {
      var s = Console.ReadLine().Split();
      if (s[0] == "insert") {
        a.AddFirst(s[1]);
      } else if (s[0] == "delete") {
        a.Remove(s[1]);
      } else if (s[0] == "deleteFirst") {
        a.RemoveFirst();
      } else {
        a.RemoveLast();
      }
    }
    Console.WriteLine(string.Join(" ", a.ToArray()));
  }
}
