import static java.lang.Integer.parseInt;
import static java.lang.Long.parseLong;
import static java.lang.System.exit;
import static java.util.Arrays.fill;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public class Main {

	static void solve() throws Exception {
		int n = scanInt();
		long k = scanLong();
		int a[] = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = scanInt();
		}
		int p[] = new int[200001];
		for (int i = n - 1; i >= 0; i--) {
			p[a[i]] = i + n;
		}
		int next[] = new int[n];
		for (int i = n - 1; i >= 0; i--) {
			next[i] = p[a[i]];
			p[a[i]] = i;
		}
		long pos[] = new long[n];
		fill(pos, -1);
		long curPos = 0, maxPos = k * n;
		int curI = 0;
		while (curPos < maxPos) {
			if (pos[curI] >= 0) {
				long jump = curPos - pos[curI];
				if (curPos + jump <= maxPos) {
					curPos += (maxPos - curPos) / jump * jump;
				}
				if (curPos >= maxPos) {
					break;
				}
			}
			pos[curI] = curPos;
			long nextPos = curPos + (next[curI] - curI + 1);
			if (nextPos > maxPos) {
				out.print(a[curI] + " ");
				++curPos;
				curI = (curI + 1) % n;
			} else {
				curPos = nextPos;
				curI = (next[curI] + 1) % n;
			}
		}
	}

	static int scanInt() throws IOException {
		return parseInt(scanString());
	}

	static long scanLong() throws IOException {
		return parseLong(scanString());
	}

	static String scanString() throws IOException {
		while (tok == null || !tok.hasMoreTokens()) {
			tok = new StringTokenizer(in.readLine());
		}
		return tok.nextToken();
	}

	static BufferedReader in;
	static PrintWriter out;
	static StringTokenizer tok;

	public static void main(String[] args) {
		try {
			in = new BufferedReader(new InputStreamReader(System.in));
			out = new PrintWriter(System.out);
			solve();
			in.close();
			out.close();
		} catch (Throwable e) {
			e.printStackTrace();
			exit(1);
		}
	}
}