using System;
class Program {
    static void Main() {
        String[] input = Console.ReadLine().Split();
        int a = int.Parse(input[0]), b = int.Parse(input[1]), c = int.Parse(input[2]);
        if (a < b && b < c) {
            Console.WriteLine("Yes");
        }
        else {
            Console.WriteLine("No");
        }
    }
}

