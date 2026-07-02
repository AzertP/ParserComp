using System;

public class ITP1_9_C{
    public static void Main(){
        var n = int.Parse(Console.ReadLine());
        
        var t_score = 0;
        var h_score = 0;
        for (var i = 0; i < n; i++)
        {
            var cards = Console.ReadLine().Split(' ');
            var t_card = cards[0];
            var h_card = cards[1];
            
            var result = String.Compare(t_card, h_card);
            if (result < 0)
            {
                h_score += 3;
            }
            else if (result == 0)
            {
                t_score++;
                h_score++;
            }
            else
            {
                t_score += 3;
            }
        }
        
        Console.WriteLine($"{t_score} {h_score}");
    }
}
