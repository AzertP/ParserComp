using System;

public class hello
{
    public static void Main()
    {
        while (true)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            if (line[1] == "?") goto readend;
            var a = int.Parse(line[0]);
            var b = int.Parse(line[2]);
            switch(line[1])
            {
                case "+":
                    Console.WriteLine(a + b);
                    break;
                case "-":
                    Console.WriteLine(a - b);
                    break;
                case "*":
                    Console.WriteLine(a * b);
                    break;
                case "/":
                    Console.WriteLine(a/ b);
                    break;
                default:
                    break;
            }
        }
        readend:;
    }
}
