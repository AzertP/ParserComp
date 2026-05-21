import java.util.*;
import java.io.*;
import java.awt.geom.*;
import java.math.*;

public class Main {

	static final BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	static final PrintWriter out = new PrintWriter(System.out,false);

	static void solve() throws IOException{
		while (true) {
			String input = br.readLine();
			if (input == null) break;

			String[] words = input.split(" ");
			boolean flag = false;
			int cip = 0;
			for (int i=0; !flag; i++) {
				cip = words[i].charAt(0) - 't';
				if (words[i].length() == 3) {
					char c1 = decipher(words[i].charAt(0)-cip);
					char c2 = decipher(words[i].charAt(1)-cip);
					char c3 = decipher(words[i].charAt(2)-cip);
					if (c1 == 't' && c2 == 'h' && c3 == 'e') {
						flag = true;
					}
				}else if (words[i].length() == 4) {
					char c1 = decipher(words[i].charAt(0)-cip);
					char c2 = decipher(words[i].charAt(1)-cip);
					char c3 = decipher(words[i].charAt(2)-cip);
					char c4 = decipher(words[i].charAt(3)-cip);
					if (c1 == 't' && c2 == 'h') {
						if ((c3 == 'i' && c4 == 's') || (c3 == 'a' && c4 == 't')){
							flag = true;
						}
					}
				}
			}

			char[] seq = input.toCharArray();
			StringBuilder sb = new StringBuilder();
			for (int i=0; i<seq.length; i++)
				sb.append((seq[i]==' ' || seq[i]=='.' || seq[i]=='\n' ? (char)seq[i] : decipher(seq[i]-cip)));
			out.println(sb);
		}
	}

	static char decipher(int x){
		x -= 'a';
		while (x < 0)
			x += 26;
		while (26 <= x)
			x -= 26;
		return (char)(x + 'a');
	}

	public static void main(String[] args) throws Exception {
		long start = System.currentTimeMillis();

		solve();
		out.flush();

		long end = System.currentTimeMillis();
		//trace(end-start + "ms");
		br.close();
		out.close();
	}

	static void trace(Object... o) { System.out.println(Arrays.deepToString(o));}
}