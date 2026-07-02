using System;
using System.Linq;
using System.Collections;
class Program {
    static void Main() {
        double R = double.Parse(Console.ReadLine());
        double PI = Math.Acos(-1);
        Console.WriteLine("{0:0.000000} {1:0.000000}", R * R * PI, 2 * R * PI);
    }
}

