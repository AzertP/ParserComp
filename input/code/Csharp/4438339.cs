using System;

public class ALDS1_1_D{
    public static void Main(){
        var n = int.Parse(Console.ReadLine());
        var tmp_min = int.Parse(Console.ReadLine());
        
        var profit = -1000000001;
        n--;
        
        for (var i = 0; i < n; i++)
        {
            var R_i = int.Parse(Console.ReadLine());
            
            if (R_i - tmp_min > profit)
            {
                profit = R_i - tmp_min;
            }
            
            if (R_i < tmp_min)
            {
                tmp_min = R_i;
            }
        }
        
        Console.WriteLine(profit);
    }
}
