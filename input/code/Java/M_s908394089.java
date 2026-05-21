
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc=new Scanner(System.in);
		int n;
		while(true){
			n=sc.nextInt();
			if(n==-1) break;
			System.out.println(f(n));
		}
	}
	static String f(int n){
		int four[]=new int[10];
		four[0]=1;
		String str="";
		for(int i=1;i<10;i++){
			four[i]=four[i-1]*4;
		}
		for(int i=9;0<=i;i--){
			if(four[i]<=n){
				str=str+Integer.toString(n/four[i]);
				n%=four[i];
			}
			else if(!str.equals("")){
				str=str+"0";
			}
		}
		if(str.equals("")){
			return "0";
		}
		else return str;
	}
}