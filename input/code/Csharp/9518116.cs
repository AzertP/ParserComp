using System;
using System.Linq;
using System.Collections;
using System.Diagnostics;
class Program {
    static void Main() {
        while (true) {
            string[] input = Console.ReadLine().Split();
            int a = int.Parse(input[0]);
            char op = char.Parse(input[1]);
            int b = int.Parse(input[2]);
            if (op == '?') break;
            if (op == '+') Console.WriteLine(a + b);
            else if (op == '-') Console.WriteLine(a - b);
            else if (op == '*') Console.WriteLine(a * b);
            else if (op == '/') Console.WriteLine(a / b);
            else Debug.Assert(false);
        }
    }
}

