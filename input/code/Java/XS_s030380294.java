import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String in1 = sc.nextLine().trim();
		double d = Double.parseDouble(in1);
		double ans = Math.sqrt(d);
		int n = (int)(ans / 10.0 * 10);
		System.out.println(n * n);
	}
}