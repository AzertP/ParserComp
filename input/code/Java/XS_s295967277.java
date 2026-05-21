import java.util.*;

class Main{
	public static void main(String args[]){
		Scanner sc = new Scanner(System.in);
		int a = sc.nextInt();
		int p = sc.nextInt();
		int ans = a*3+p;
		ans = ans/2;
		System.out.print(ans);
	}
}
