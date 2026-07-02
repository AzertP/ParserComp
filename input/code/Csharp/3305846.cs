using System;
using System.Collections.Generic;
using System.Text;

class Program {
  static void Main() {
    int n = int.Parse(Console.ReadLine());
    var h = new HashSet<string>();
    var sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      var s = Console.ReadLine().Split();
      if(s[0][0] == 'i') h.Add(s[1]);
      else if (h.Contains(s[1])) sb.AppendLine("yes");
      else  sb.AppendLine("no");
    }
    Console.Write(sb);
  }
}
