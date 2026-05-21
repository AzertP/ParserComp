import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Scanner;
import java.util.Set;

public class Main {

	public static void main(String[] args) {
		execute14_1();
	}

	private static void execute14_2() {
		try (Scanner sc = new Scanner(System.in)) {
			int n = sc.nextInt();
			int [] a = new int[n];
			
			for(int i=0; i<n; i++) {
				a[i] = sc.nextInt();
			}
			
			int ans =0;
			for(int i=0; i<n-1; i++) {
				for(int j=i+1; j<n; j++) {
					if(j-i==a[i]+a[j]) ans++;
				}
			}
			System.out.println(ans);
		}

	}

	private static void execute14_1() {
		try (Scanner sc = new Scanner(System.in)) {
			long x = sc.nextInt();

			int ans_a = 9999;
			int ans_b = 9999;
			if(x==1) {
				ans_a=1;
				ans_b=0;
			}
			for(int i=2; i <= x; i++) {
				long tmp = x;
				int wr = (int) ((int) tmp/Math.pow(i, 5));
				if(wr>1) continue;
				tmp%=Math.pow(i, 5);
		
				for(int n=0; n < i; n++) {
					if(tmp==Math.pow(n, 5)) {
						ans_a = i;
						ans_b = n*-1;
					} else if(Math.abs(tmp-Math.pow(i, 5))==Math.pow(n, 5)) {
						ans_a = i;
						ans_b = n;
					}
				}
				if(ans_a!=9999) break;
				
				
				
			}
			System.out.println(ans_a + " " + ans_b);
		}

	}
}