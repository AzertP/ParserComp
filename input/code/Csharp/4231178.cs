using System;
using System.Linq;
using System.Collections.Generic;

class Program {
    static void Main() {
        string[] stdin = Console.ReadLine().Split();
        int r = int.Parse(stdin[0]);
        int c = int.Parse(stdin[1]);
        List<List<int>> sheet = new List<List<int>>();
        for(int i = 0; i < r; i++) {
            sheet.Add(Console.ReadLine().Split().Select(s => int.Parse(s)).ToList());
            sheet[i].Add(sheet[i].Sum());
            Console.WriteLine(string.Join(" ", sheet[i]));
        }
        for(int i = 0; i <= c; i++) {
            int ans = 0;
            for(int j = 0; j < r; j++) ans += sheet[j][i];
            Console.Write("{0}{1}", ans, i == c? '\n' : ' ');
        }
    }
}
