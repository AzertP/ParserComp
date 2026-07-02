using System;

public class Test
{
	public static void Main()
	{
		var arrnum = Int32.Parse(Console.ReadLine().Trim());
		var arr =Array.ConvertAll(Console.ReadLine().Split(' '), Int32.Parse);
	
		Console.WriteLine(findMinMaxSum(arrnum, arr));
	}
	
	public static string findMinMaxSum(int arrnum, int[] arr){
		int min=int.MaxValue, max = int.MinValue;
		long sum=0;
		
		for( int i=0; i<arrnum; i++)
		{
			if(min > arr[i])
				min = arr[i];
			
			 if (max < arr[i])
				max = arr[i];
			
			sum+=arr[i];
		}
		return min +" "+ max +" "+ sum;
	}
}
