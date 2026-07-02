using System;
using System.Linq;

public class ITP1_9_D{
    public static void Main(){
        var str = Console.ReadLine();
        var q = int.Parse(Console.ReadLine());
        
        for (var i = 0; i < q; i++)
        {
            var command = Console.ReadLine().Split(' ');
            var cmd_name = command[0];
            var a = int.Parse(command[1]);
            var b = int.Parse(command[2]);
            
            var len = b - a + 1;
            
            if (cmd_name == "print")
            {
                Console.WriteLine(str.Substring(a, len));
            }
            else
            {
                var substr1 = str.Substring(0, a);
                var substr2 = str.Substring(b + 1);
                
                if (cmd_name == "reverse")
                {
                    var substr_rev = str.Substring(a, len);
                    substr_rev = string.Join("", substr_rev.Reverse());
                    str = substr1 + substr_rev + substr2;
                }
                else
                {
                    str = substr1 + command[3] + substr2;
                }
            }
        }
    }
}
