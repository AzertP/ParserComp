using System;
using System.Linq;

public class ALDS1_1_A{
    public static void Main(){
        var N = int.Parse(Console.ReadLine());
        var A = Console.ReadLine().Split(' ').Select(int.Parse).ToArray();
        
        for (var i = 1; i < N; i++)
        {
            Console.WriteLine(string.Join(" ", A));
            
            var key = A[i];
            var j = i - 1;
            while (j >= 0 && A[j] > key)
            {
                A[j+1] = A[j];
                j--;
            }
            A[j+1] = key;
        }
        
        Console.WriteLine(string.Join(" ", A));
    }
}

