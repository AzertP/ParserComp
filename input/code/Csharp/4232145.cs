using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        string x;
        while((x = Console.ReadLine()) != "0") {
            int ans = 0;
            for(int i = 0; i < x.Length; i++) ans += int.Parse(x.Substring(i, 1));
            Console.WriteLine(ans);
        }
    }
}
