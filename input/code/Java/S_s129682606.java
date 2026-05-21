import java.util.*;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		int[] a = new int[n];
		for (int i = 0; i < a.length; i++) {
			a[i] = sc.nextInt();
		}
		int count = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] %2 != 0) {
				continue;
			} else {
				while (true) {
					if (a[i]%2==0) {
						count++;
						a[i] /= 2;
					} else {
						break;
					}
				}
			}
		}
		System.out.println(count);
		
		sc.close();
	}
}
