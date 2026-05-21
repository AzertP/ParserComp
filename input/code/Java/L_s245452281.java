import java.util.Random;
import java.util.Scanner;

public class Main {
	private static final int K = 26;
	private static Random rand = new Random(System.currentTimeMillis());

	public static void main(String[] args) {
		var sc = new Scanner(System.in);
		var d = sc.nextInt();
		var c = new int[K];
		var s = new int[d][K];
		for (var i = 0; i < K; i++) {
			c[i] = sc.nextInt();
		}
		for (var i = 0; i < d; i++) {
			for (var j = 0; j < K; j++) {
				s[i][j] = sc.nextInt();
			}
		}
		sc.close();

		var t = create(c, s, d);
		var max = score(c, s, t, d);
		for (var i = 0; i < 1000; i++) {
			for (var j = 1; j < 200; j++) {
				var index = i % d;
				var t2 = new int[d];
				for (var k = 0; k < d; k++) {
					t2[k] = t[k];
				}
				t2[index] = (t2[index] + j) % K + 1;
				var sum = score(c, s, t2, d);
				//System.out.println(sum);
				if (max < sum) {
					t = t2;
					max = sum;
				}
			}
		}

		for (var i = 0; i < d; i++) {
			System.out.println(t[i]);
		}
	}

	private static int score(int[] c, int[][] s, int[] t, int d) {
		var sum = 0;
		var last = new int[K];
		for (var i = 0; i < d; i++) {
			var ti = t[i] - 1;
			sum += s[i][ti];
			last[ti] = i + 1;
			for (var j = 0; j < K; j++) {
				sum -= ((i + 1) - last[j]) * c[j];
			}
		}

		return sum;
	}

	private static int[] create(int[] c, int[][] s, int d) {
		var t = new int[d];
		var last = new int[K];
		for (var i = 0; i < d; i++) {
			var v = new int[K];
			for (var j = 0; j < K; j++) {
				v[j] = s[i][j] + (((i + 1) - last[j]) * c[j]);
			}
			t[i] = rand(v) + 1;
			last[t[i] - 1] = i + 1;
		}
		return t;
	}

	private static int rand(int[] v) {
		var sum = 0;
		for (var i = 0; i < v.length; i++) {
			sum += v[i];
		}
		var value = rand.nextInt(sum);
		var index = 0;
		sum = 0;
		for (var i = 0; i < v.length; i++) {
			sum += v[i];
			if (value < sum) {
				break;
			}
			index++;
		}
		//System.out.printf("%d %d %d\r\n", value, sum, index);
		return index;
	}
}
