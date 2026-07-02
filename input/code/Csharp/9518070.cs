using System;
class Program {
    static void Main() {
        string[] input = Console.ReadLine().Split();
        int H = int.Parse(input[0]);
        int W = int.Parse(input[1]);
        Console.WriteLine("{0} {1}", H * W, 2 * (H + W));
    }
}

