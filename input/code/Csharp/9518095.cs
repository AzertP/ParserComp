using System;
using System.Linq;
using System.Collections;
class Program {
    static void Main() {
        while (true) {
            String[] input = Console.ReadLine().Split();
            int x = int.Parse(input[0]), y = int.Parse(input[1]);
            if (x == 0 && y == 0) break;
            if (x > y) {
                (y, x) = (x, y);
            }
            Console.WriteLine("{0} {1}", x, y);
        }
    }
}

