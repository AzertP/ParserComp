using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        string w = Console.ReadLine();
        string[] t;
        int ans = 0;
        while((t = Console.ReadLine().Split().ToArray())[0] != "END_OF_TEXT") {
            ans += t.Count(s => string.Compare(s, w, true) == 0);
        }
        Console.WriteLine(ans);
    }
}
