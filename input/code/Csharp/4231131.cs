using System;
using System.Linq;

class Program {
    static void Main() {
        while(true) {
            string[] stdin = Console.ReadLine().Split();
            int m = int.Parse(stdin[0]);
            int f = int.Parse(stdin[1]);
            int r = int.Parse(stdin[2]);
            if(m == -1 && f == -1 && r == -1) break;
            char ans;
            if(m == -1 || f == -1) ans = 'F';
            else if(m + f >= 80) ans = 'A';
            else if(m + f >= 65) ans = 'B';
            else if(m + f >= 50) ans = 'C';
            else if(m + f >= 30) {
                if(r >= 50) ans = 'C';
                else ans = 'D';
            }else ans = 'F';
            Console.WriteLine(ans);
        }
    }
}
