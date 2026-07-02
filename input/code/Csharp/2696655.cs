using System;
using System.Collections.Generic;

public static class Program
{
	public static void Main()
	{
		var stack = new Stack<int>();
		foreach (var s in Console.ReadLine().Split(' '))
		{
			switch (s)
			{
			case "+":
				stack.Push(stack.Pop() + stack.Pop());
				break;
			case "-":
				int a = stack.Pop();
				int b = stack.Pop();
				stack.Push(b - a);
				break;
			case "*":
				stack.Push(stack.Pop() * stack.Pop());
				break;
			default:
				stack.Push(int.Parse(s));
				break;
			}
		}
		Console.WriteLine(stack.Pop());
	}
}

