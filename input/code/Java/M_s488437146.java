
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		while (true) {
			int n = scanner.nextInt();
			if (n == 0) {
				break;
			}
			int s = scanner.nextInt();
			int g = scanner.nextInt();
			int d = scanner.nextInt();
			boolean[][] map = new boolean[d + 1][n + 1];
			for (int i = 1; i <= d; i++) {
				String line = scanner.next();
				for (int j = 1; j < n; j++) {
					if (line.charAt(j - 1) == '1') {
						map[i][j] = true;
					}
				}
			}
			boolean flag = false;
			loop:
			for (int i = 0; i <= d; i++) {
				for (int j = 1; j < n; j++) {
					if (!map[i][j - 1] && !map[i][j] && !map[i][j + 1]) {
						int x = s;
						map[i][j] = true;
						for (int k = 1; k <= d; k++) {
							if (map[k][x - 1]) {
								x--;
							} else if (map[k][x]) {
								x++;
							}
						}
						if (g == x) {
							if (i == 0) {
								System.out.println(0);
							} else {
								System.out.println(i + " " + j);
							}
							flag = true;
							break loop;
						}
						map[i][j] = false;
					}

				}
			}if(!flag){
				System.out.println(1);
			}
		}
	}
}