using System;
using System.Linq;

internal class AizuOnlineJudge {
    public static void Main() {
        var max = int.Parse(Console.ReadLine());

        var bubble = Console.ReadLine().Split();
        var selection = bubble.ToArray();

        for (var i = 0; i < max; i++) {
            for (var j = 0; j < max - 1; j++) {
                if (bubble[j][1] > bubble[j + 1][1]) {
                    var temp = bubble[j];
                    bubble[j] = bubble[j + 1];
                    bubble[j + 1] = temp;
                }
            }
        }

        for (var i = 0; i < max; i++) {
            var min = i;
            for (var j = i + 1; j < max; j++) {
                if (selection[j][1] < selection[min][1]) {
                    min = j;
                }
            }

            var temp = selection[i];
            selection[i] = selection[min];
            selection[min] = temp;
        }

        Console.WriteLine(string.Join(" ", bubble));
        Console.WriteLine("Stable");
        Console.WriteLine(string.Join(" ", selection));
        Console.WriteLine(bubble.SequenceEqual(selection) ? "Stable" : "Not stable");
    }
}

