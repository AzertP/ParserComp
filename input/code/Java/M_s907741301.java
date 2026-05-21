import java.util.ArrayDeque;
import java.util.Scanner;

public class Main {

	static char	s[];
	static char	t[];

	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		s = in.next().toCharArray();
		t = in.next().toCharArray();
		in.close();

		int row = t.length;
		int col = s.length;
		int tb[][] = new int[row + 1][col + 1];
		for(int i = 0; i<=row; i++)tb[i][0] = 0;
		for(int i = 0; i<=col; i++)tb[0][i] = 0;

		for ( int r = 0; r < row; r++ ) {
			for ( int c = 0; c < col; c++ ) {
				if ( t[r] == s[c] ) {
					tb[r + 1][c + 1] = tb[r][c] + 1;
				} else {
					tb[r + 1][c + 1] = Math.max(tb[r + 1][c], tb[r][c + 1]);
				}
			}
		}

		//		for ( int[] is : tb ) {
		//			for ( int i : is ) {
		//				System.out.print(i + " ");
		//			}
		//			System.out.println();
		//		}

		int r = row;
		int c = col;
		ArrayDeque<Character> stack = new ArrayDeque<Character>();
		while ( tb[r][c] != 0 ) {

			// 左上に行けるとき
			if ( tb[r - 1][c - 1] == tb[r][c]) {
				r -= 1;
				c -= 1;
			} else
			// 上にだけ行けるとき
			if ( tb[r - 1][c] ==  tb[r][c]) {
				r -= 1;
			} else
			// 左にだけ行けるとき
			if ( tb[r][c - 1] == tb[r][c]) {
				c -= 1;
			} else
			// 行き止まり
			{
				stack.add((Character) s[c - 1]);
				r -= 1;
				c -= 1;
			}
		}
		if ( stack.size() == 0 ) {
			System.out.print("");
			return;
		}

		while ( stack.size() > 0 ) {
			System.out.print(stack.pollLast());
		}
	}
}
