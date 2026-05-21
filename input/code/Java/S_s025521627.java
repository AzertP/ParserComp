import java.util.Scanner;

public class Main {

	@SuppressWarnings("resource")
	public static void main(String args[]) {
		Scanner scanner = new Scanner(System.in);
		int n = scanner.nextInt();
		int d = scanner.nextInt();
		int x = scanner.nextInt();

		int co = 0;
		for (int i = 0; i < n; i++) {
			co += eat(scanner.nextInt(), d);
		}

		System.out.println(x + co);
	}

	private static int eat(int t, int d) {
		int co = 0;
		for (int i = 1; i <= d; i += t) {
			co++;
		}
		return co;
	}
}
