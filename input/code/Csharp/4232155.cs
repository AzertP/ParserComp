using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        int[] ans = new int[26];
        string str;
        while((str = Console.ReadLine()) != null) {
            foreach (char c in str) {
                for(int i = 0; i < 26; i++) {
                    if(c == 'a' + i || c == 'A' + i) ans[i]++;
                }
            }
        }
        for(int i = 0; i < 26; i++) Console.WriteLine("{0} : {1}", (char)('a' + i), ans[i]);
    }
}
