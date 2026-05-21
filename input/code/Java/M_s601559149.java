

import java.util.Scanner;

public class Main {


		public static void main(String[] agrs)
		{
			Scanner sc = new Scanner(System.in);
			int n = sc.nextInt();
			int l = sc.nextInt();
			int a=2;
			int num=0;
			int min=l;
			while(a<=n)
			{
				num=l+a-1;
				if(Math.abs(num)<Math.abs(min))
				{
					min = num;
				}
				++a;
			}
		
			int sum=0;
			int b=1;
			while(b<=n)
			{
				num=l+b-1;
				while(num!=min)
				{
					sum+=num;
					break;
				}
				++b;
			}
			System.out.println(sum);
			
		}
		}
	 
	

