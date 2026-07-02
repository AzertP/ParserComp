using System;
using System.Linq;
using System.Collections;
using System.Diagnostics;
using System.Collections.Generic;
class Program {
    static void Main() {
        int N = int.Parse(Console.ReadLine());
        var set = new HashSet<string>();
        for (int i = 0 ; i < N ; i++) {
            set.Add(Console.ReadLine());
        }
        string[] suit = {"S", "H", "C", "D"};
        foreach(string s in suit) {
            for (int i = 1 ; i <= 13 ; i++) {
                string cur = s + " " + i.ToString();
                if (!set.Contains(cur)) {
                    Console.WriteLine(cur);
                }
            }
        }
    }
}

