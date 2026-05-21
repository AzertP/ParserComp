import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String s = sc.nextLine();
		int n = Integer.parseInt(s);
		int fx = 0;
		for (String tmp : s.split("")) {
			fx += Integer.parseInt(tmp);
		}

		if (n % fx == 0) {
			System.out.println("Yes");
		} else {
			System.out.println("No");
		}

		sc.close();
	}
}