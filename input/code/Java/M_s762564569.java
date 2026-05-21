import java.awt.geom.Point2D;
import java.util.Scanner;

public class Main {
	public static void main(String args[]) {
		Scanner sc = new Scanner(System.in);
		Point2D.Double p[];
		String str[];
		
		p = new Point2D.Double[4];
		while (sc.hasNext()) {
			str = sc.nextLine().split(" ");
			for (int i = 0; i < 4; i++) {
				p[i] = new Point2D.Double(Double.valueOf(str[2 * i]), Double.valueOf(str[2 * i + 1]));
			}
			
			if (p[0].x <= p[3].x && p[2].x <= p[1].x) {
				if (p[0].y <= p[3].y && p[2].y <= p[1].y) {
					System.out.println("YES");
				} else if (p[2].y <= p[1].y && p[0].y <= p[3].y) {
					System.out.println("YES");
				} else {
					System.out.println("NO");
				}
			} else if (p[2].x <= p[1].x && p[0].x <= p[3].x) {
				if (p[0].y <= p[3].y && p[2].y <= p[1].y) {
					System.out.println("YES");
				} else if (p[2].y <= p[1].y && p[0].y <= p[3].y) {
					System.out.println("YES");
				} else {
					System.out.println("NO");
				}
			} else {
				System.out.println("NO");
			}
		}
	}
}