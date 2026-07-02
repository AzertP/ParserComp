using System;
using System.Linq;
using System.Collections;
class Program {

    static int read() {
        int res = int.Parse(Console.ReadLine());
        return res;
    }
    static void Main() {
        for (int i = 1 ; ; i++) {
            int x = read();
            if (x == 0) break;
            Console.WriteLine("Case {0}: {1}", i, x);
        }
    }
}

