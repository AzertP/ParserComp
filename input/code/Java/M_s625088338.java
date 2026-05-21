
import java.util.*;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class Main {
	int N;
	int result = -1;

	public static void main(String[] args) throws IOException {
		InputStreamReader reader = new InputStreamReader(System.in, StandardCharsets.UTF_8);
		BufferedReader in = new BufferedReader(reader);
		Main ins = new Main(in);
		ins.calc();
		ins.showResult();
	}

	Main(BufferedReader in) throws IOException {
		N = Integer.parseInt(in.readLine());
	}

	void calc() {
		int val = (int) Math.floor((double) N / 1.08);
		for (int i = val - 1; i <= val + 1; ++i) {
			int v = (int) Math.floor((double) i * 1.08);
			if (v == N) {
				result = i;
				break;
			}
		}
	}

	void showResult() {
		if (result < 0) {
			System.out.println(":(");
		} else {
			System.out.println(result);
		}
	}
}
