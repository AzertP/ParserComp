using System;

public class Test
{
	public static void Main()
	{
		// your code goes here
		
		string s=Console.ReadLine();
		string [] s1 = s.Split(' ');
		double c;
		long a= Convert.ToInt64(s1[0].ToString());
		long b= Convert.ToInt64(s1[1].ToString());
		 
		 if(a>=1 && a<=Math.Pow(10,9) && b>=1 && b<=Math.Pow(10,9)) {
		  c =(double)a/(double)b;
		  Console.WriteLine("{0} {1} {2:0.00000} ",(a/b),a%b,c);
		 }
	}
}
