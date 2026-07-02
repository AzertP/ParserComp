using System;

public class Test
{
	public static void Main()
	{
		while(true)
		{
		int first;
		int second;
		string[] nums = Console.ReadLine().Split(' ');
	    first = Int32.Parse(nums[0]);
        second =  Int32.Parse(nums[1]);
        if(first == 0 && first==second)
        {
        	break;
        }
        else
        {
        	if(first > second)
        	{
        		Console.WriteLine(second + " " + first);
        	}
        	else
        	{
        			Console.WriteLine(first + " " + second);
        	}
        }
    }
}
}
