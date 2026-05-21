import java.util.Scanner;

public class Main{

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		Twin_string ts = new Twin_string(sc.next());
		int q = sc.nextInt();
		int t, f;
		boolean reverse = false;
		for (int i = 0; i < q; i++) {
			t = sc.nextInt();
			if(t==1) {
				reverse = !reverse;
			}else {
				f = sc.nextInt();
				String c = sc.next();
				if((!reverse&&f==1)||(reverse&&f==2)) ts.addFirst(c);
				else ts.addLast(c);;
			}
		}
		System.out.println(ts.toString(reverse));
		sc.close();
	}
	public static class Twin_string{
		StringBuilder sb1;
		StringBuilder sb2;
		public Twin_string(String s) {
			sb1 = new StringBuilder();
			sb2 = new StringBuilder(s);
		}
		public void addFirst(String s) {
			sb1.append(s);
		}
		public void addLast(String s) {
			sb2.append(s);
		}
		public String toString(boolean reverse) {
			if(reverse) {
				return new String(sb2.reverse()) + new String(sb1);
			}else {
				return new String(sb1.reverse()) + new String(sb2);
			}
		}
	}
}