import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int number = sc.nextInt();
		int num = sc.nextInt();

		int res = (number+1) / 2;

		if(res >= num) {
			System.out.println("YES");
		}else {
			System.out.println("NO");
		}
	}
}