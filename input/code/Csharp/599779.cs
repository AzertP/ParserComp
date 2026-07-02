using System;
using System.Collections.Generic;

class Program {
  static void Main(string[] args) {
    string line = Console.ReadLine();
    string[] elem = line.Split(' ');
    var stack = new Stack<string>();
    foreach (var e in elem) {
      int num;
      if (int.TryParse(e, out num)) {
        stack.Push(e);
      }
      else {
        int right = int.Parse(stack.Pop());
        int left  = int.Parse(stack.Pop());
        if(e == "+") {
          int ans = left + right;
          stack.Push(ans.ToString());
        }
        else if(e == "-") {
          int ans = left - right;
          stack.Push(ans.ToString());
        }
        else if(e == "*") {
          int ans = left * right;
          stack.Push(ans.ToString());
        }
      }
    }
    Console.WriteLine(stack.Pop());
  }
}
