using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        string str;
        while((str = Console.ReadLine()) != "-") {
            int m = int.Parse(Console.ReadLine());
            for(int i = 0; i < m; i++) {
                int h = int.Parse(Console.ReadLine());
                str = str.Substring(h) + str.Substring(0, h);
            }
            Console.WriteLine(str);
        }
    }
}
