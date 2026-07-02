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
  }
}

class Program {
  static void Main() {
    Dice dice = new Dice(Console.ReadLine().Split().Select(int.Parse).ToArray());
    var dir = "NNNENNN";
    int q = int.Parse(Console.ReadLine());
    while (q-- > 0) {
      var a = Console.ReadLine().Split().Select(int.Parse).ToArray();
      foreach (var c in dir) {
        if (dice.label[1] == a[1]) break;
        dice.Roll(c);
      }
      while (dice.label[0] != a[0]) {
        dice.Roll('E');
      }
      Console.WriteLine(dice.label[2]);
    }
  }
}
