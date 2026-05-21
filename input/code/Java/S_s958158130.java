import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class Main {

	static int dp[][] = new int[(1 << 21) + 1][21];
	static int dist[][] = new int[21][21];

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		int a = Integer.parseInt(br.readLine());

		System.out.println((int) (a + Math.pow(a, 2) + Math.pow(a, 3)));
	}
}
