import java.util.Scanner;

public class Main {
	private double t, y;

	public static void main(String[] args) {
		new Main().run();
	}
	
	private void run() {
		Scanner scan = new Scanner(System.in);
		while (scan.hasNext()) {
			double v = scan.nextDouble();
			
			t = v / 9.8;
			y = 4.9 * t * t;
			
			int N = 1;
			while (true) {
				if (5 * N - 5 >= y) break;
				N++;
			}
			//debug();
			System.out.println(N);
		}
	}
	
	private void debug() {
		System.out.println(t + " " + y);
	}
}