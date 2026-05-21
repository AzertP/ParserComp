import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		int k = sc.nextInt() - 1;
		char[] c = sc.next().toCharArray();
		c[k] += 32;
		System.out.println(new String(c));
		sc.close();
	}
}
