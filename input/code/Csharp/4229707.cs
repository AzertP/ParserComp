using System;
using System.Linq;

class Program {
    static void Main() {
        while(true) {
            string[] stdin = Console.ReadLine().Split();
            int h = int.Parse(stdin[0]);
            int w = int.Parse(stdin[1]);
            if((h | w) == 0) break;
            for(int i = 0; i < h; i++) {
                for(int j = 0; j < w; j++) Console.Write(i == 0 || i == h - 1 || j == 0 || j == w - 1? "#" : ".");
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}
