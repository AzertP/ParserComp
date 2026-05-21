
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		//		new Main().solveA();
		new Main().solveB();
		// new Main().solveC();
		// new Main().solveD();
	}

	private void solveA() {
		Scanner scanner = null;
		String wk = "";

		try {
			scanner = new Scanner(System.in);
			wk = scanner.next();

			System.out.println(wk.replaceAll(",", " "));

		} finally {
			if (scanner != null) {
				scanner.close();
			}
		}
	}

	private void solveB() {
		Scanner scanner = null;
		int numK = 0;
		int s = 0;

		try {
			scanner = new Scanner(System.in);
			numK = scanner.nextInt();
			s = scanner.nextInt();

			int startK = 0;
			if (numK <= s / 3) {
				startK = numK;
			}
			int res = 0;
			for (int i = 0; i <= numK; i++) {
				for (int j = 0; j <= numK; j++) {
					if (s - i - j < 0) {
						break;
					} else if (s - i - j <= numK) {
						res++;
					}
					//					startK = s - i - j;
					//					for (int k = startK; k <= numK; k++) {
					//						if (i + j + k == s) {
					//							res++;
					//							break;
					//						}
					//					}
				}
			}

			System.out.println(res);

		} finally {
			if (scanner != null) {
				scanner.close();
			}
		}
	}

	private void solveC() {
		Scanner scanner = null;
		int lineAB = 0;
		int lineBC = 0;
		int lineCA = 0;

		try {
			scanner = new Scanner(System.in);
			lineAB = scanner.nextInt();

			System.out.println("");

		} finally {
			if (scanner != null) {
				scanner.close();
			}
		}
	}

	private void solveD() {
		Scanner scanner = null;
		int lineAB = 0;
		int lineBC = 0;
		int lineCA = 0;

		try {
			scanner = new Scanner(System.in);
			lineAB = scanner.nextInt();

			System.out.println("");

		} finally {
			if (scanner != null) {
				scanner.close();
			}
		}
	}
}