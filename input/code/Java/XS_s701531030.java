import java.util.Scanner;
public class Main{
	public static void main(String[]args) {
		Scanner input=new Scanner(System.in);
		int a;
		int b;
		int h;
		do {
			a=input.nextInt();
			b=input.nextInt();
			h=input.nextInt();
		}while(a<=1&&a>100&&b<=1&&b>100&&h<=1&&h>100&&h%2!=0);
		System.out.println((a+b)*h/2);
	}
}