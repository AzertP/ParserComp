import java.util.ArrayList;
import java.util.Scanner;

public class Main {
	void run() {
		Scanner scan = new Scanner(System.in);
		ArrayList<Integer> a = new ArrayList<Integer>();
		ArrayList<String> op = new ArrayList<String>();
		ArrayList<Integer> b = new ArrayList<Integer>();
		ArrayList<Integer> ans = new ArrayList<Integer>();
		
		do {
			a.add(scan.nextInt());
			op.add(scan.next());
			b.add(scan.nextInt());
		} while (!op.contains("?"));
		for (int i = 0; i < op.size(); i++) {
			int _a = a.get(i);
			int _b = b.get(i);
			if (op.get(i).equals("+")) {
				ans.add(_a + _b);
			} else if (op.get(i).equals("-")) {
				ans.add(_a - _b);
			} else if (op.get(i).equals("*")) {
				ans.add(_a * _b);
			} else if (op.get(i).equals("/")) {
				ans.add(_a / _b);
			} else break;
			System.out.println(ans.get(i));
		}
	}
	public static void main(String[] args) {
		Main m = new Main();
		m.run();
	}
}