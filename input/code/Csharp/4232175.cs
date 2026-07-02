using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        string s = Console.ReadLine();
        string t = Console.ReadLine();
        s += s;
        for(int i = 0; i + t.Length <= s.Length; i++) {
            if(s.Substring(i, t.Length) == t) {
                Console.WriteLine("Yes");
                return;
            }
        }
        Console.WriteLine("No");
    }
}
