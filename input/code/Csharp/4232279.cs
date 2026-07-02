using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        int a = 0, b = 0;
        int n = int.Parse(Console.ReadLine());
        for(int i = 0; i < n; i++) {
            string[] stdin = Console.ReadLine().Split();
            if(stdin[0].CompareTo(stdin[1]) < 0) b += 3;
            else if(stdin[0].CompareTo(stdin[1]) > 0) a += 3;
            else {
                a++;
                b++;
            }
        }
        Console.WriteLine("{0} {1}", a, b);
    }
}
