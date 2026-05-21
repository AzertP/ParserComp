
import java.util.Arrays;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int n = Integer.parseInt(sc.next());
		
		int []a = new int [n];
		
		for(int i = 0 ; i < n ;i++) {
			a[i] = Integer.parseInt(sc.next());
		}
		
		Arrays.sort(a);
		
		long ans = a[n-1];
		int id = n-2;
		int cnt = n-2;
		
		while(cnt > 0) {
			if(cnt >= 2) {
				ans += (long)(a[id]*2);
				cnt-=2;
			}
			else {
				ans += (long)a[id];
				cnt--;
			}
			id--;
		}
		
		System.out.println(ans);
		
		
		
	}
}
