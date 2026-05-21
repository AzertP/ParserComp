import java.io.PrintStream;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = null;
		PrintStream out = null;
		try {
			sc = new Scanner(System.in);
			out = System.out;
			exec(sc, out);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		} finally {
			if(out != null) {
				out.close();
			}
			if(sc != null) {
				sc.close();
			}
		}
		System.exit(0);
	}
	public static void exec(final Scanner sc, final PrintStream out) throws Exception{
		final int N = sc.nextInt();
		final int K = sc.nextInt();
		final String S = sc.next();
		final boolean[] T = new boolean[N];
		for(int i=0;i<S.length();i++) {
			T[i] = S.charAt(i)=='L';
		}

		boolean base = T[0];
		int w = 1;
		for(int k=0;k<K;k++) {
			int l=w;
			for(;l<N;l++) {
				if(T[l]!=base) {
					break;
				}
			}
			if(l==N) {
				break;
			}
			int r=l+1;
			for(;r<N;r++) {
				if(T[r]==base) {
					break;
				}
			}
			r--;
			// 180度回転
			while(l<r) {
				boolean t = T[r];
				T[r] = !T[l];
				T[l] = !t;
				l++;
				r--;
			}
			if(l==r) {
				T[l]=!T[r];
			}
			w = r+1;
		}

		int score = 0;
		for(int i=0;i<N-1;i++) {
			if(T[i]==T[i+1]) {
				score++;
			}
		}

		out.println(score);
	}

}
