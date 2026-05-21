import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(System.in);
		int a = Integer.parseInt(scanner.nextLine());
		String s = scanner.nextLine();
		if (a >= 3200) {
			System.out.println(s);
		} else {
			System.out.println("red");
		}
	}
}