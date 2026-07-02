using System;
using System.Linq;

class Program {
    static void Main() {
        int n = int.Parse(Console.ReadLine());
        for(int i = 1; i <= n; i++) {
            if(i % 3 == 0) {
                Console.Write(" {0}", i);
                continue;
            }
            int x = i;
            while(x > 0) {
                if(x % 10 == 3) {
                    Console.Write(" {0}", i);
                    break;
                }
                x /= 10;
            }
        }
        Console.WriteLine();
    }
}
