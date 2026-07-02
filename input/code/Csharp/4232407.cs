using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        string str = Console.ReadLine();
        int q = int.Parse(Console.ReadLine());
        for(int i = 0; i < q; i++) {
            string[] cmd = Console.ReadLine().Split();
            int a = int.Parse(cmd[1]);
            int b = int.Parse(cmd[2]);
            if(cmd[0] == "print") Console.WriteLine(str.Substring(a, b - a + 1));
            else if(cmd[0] == "reverse") str = str.Substring(0, a) + string.Join("", str.Substring(a, b - a + 1).Reverse()) + str.Substring(b + 1);
            else str = str.Substring(0, a) + cmd[3] + str.Substring(b + 1);
        }
    }
}
