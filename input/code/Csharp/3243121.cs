using System;
using System.Linq;
using System.Collections.Generic;

class Dice {
  public int[] label;
  public Dice(int[] a) {
    label = a;
  }
  public void Swap(int a, int b, int c, int d) {
    int temp = label[a];
    label[a] = label[b];
    label[b] = label[c];
    label[c] = label[d];
    label[d] = temp;
  }
  public void Roll(char dir) {
    if (dir == 'N') Swap(0, 1, 5, 4);
    else if (dir == 'E') Swap(0, 3, 5, 2);
    else if (dir == 'W') Swap(0, 2, 5, 3);
    else if (dir == 'S') Swap(0, 4, 5, 1);
    else if (dir == 'R') Swap(1, 2, 4, 3);
  }
  public bool IsSame(Dice dice) {
    foreach (var dir in "RRRRNRRRRERRRRNRRRRERRRRNRRRR") {
      if (label.SequenceEqual(dice.label)) return true;
      dice.Roll(dir);
    }
    return false;
  }
}

class Program {
  static void Main() {
    int n = int.Parse(Console.ReadLine());
    List<Dice> dice = new List<Dice>();
    for (int i = 0; i < n; i++) {
      dice.Add(new Dice(Console.ReadLine().Split().Select(int.Parse).ToArray()));
    }
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if (dice[i].IsSame(dice[j])) {
          Console.WriteLine("No");
          return;
        }
      }
    }
    Console.WriteLine("Yes");
  }
}
