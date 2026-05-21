
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		new Main().run();
	}

	public void run() {
		Scanner sc = new Scanner(System.in);
		out: while (true) {
			int[] h = new int[2];
			int[] w = new int[2];
			char[][][] p = new char[2][][];
			for (int t = 0; t < 2; ++t) {
				h[t] = sc.nextInt() + 2;
				w[t] = sc.nextInt() + 2;
				if (h[t] == 2 && w[t] == 2)
					break out;
				p[t] = new char[h[t]][w[t]];
				for (int i = 1; i < h[t] - 1; ++i) {
					p[t][i] = ("." + sc.next() + ".").toCharArray();
				}
				Arrays.fill(p[t][0], '.');
				Arrays.fill(p[t][h[t] - 1], '.');
			}
			solve(h, w, p);
		}
	}

	void solve(int[] h, int[] w, char[][][] p) {
		Tree[] tree = new Tree[2];
		for (int t = 0; t < 2; ++t) {
			tree[t] = f(h[t], w[t], p[t]);
		}
		boolean flag = tree[0].hash() == tree[1].hash();
		System.out.println(flag ? "yes" : "no");
	}

	Tree f(int h, int w, char[][] p) {
		DJSet ds = new DJSet(h * w);
		@SuppressWarnings("unchecked")
		ArrayList<Integer>[] g = new ArrayList[h * w];
		for (int i = 0; i < g.length; ++i) {
			g[i] = new ArrayList<>();
		}
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				for (int dx = -1; dx <= 1; ++dx) {
					for (int dy = -1; dy <= 1; ++dy) {
						if (dx == 0 && dy == 0)
							continue;
						if (i + dy < 0 || j + dx < 0 || i + dy >= h || j + dx >= w)
							continue;
						if (p[i][j] == '.' && dx * dy != 0)
							continue;
						if (p[i + dy][j + dx] == p[i][j]) {
							ds.setUnion(i * w + j, (i + dy) * w + (j + dx));
						}
					}
				}
			}
		}
		DJSet ds2 = new DJSet(h * w);
		ds2.upper = Arrays.copyOf(ds.upper, ds.upper.length);
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				for (int dx = -1; dx <= 1; ++dx) {
					for (int dy = -1; dy <= 1; ++dy) {
						if (dx == 0 && dy == 0)
							continue;
						if (dx * dy != 0)
							continue;
						if (i + dy < 0 || j + dx < 0 || i + dy >= h || j + dx >= w)
							continue;
						int v = ds.root(i * w + j);
						int u = ds.root((i + dy) * w + (j + dx));
						if (!ds2.equiv(u, v)) {
							g[v].add(u);
							g[u].add(v);
							ds2.setUnion(u, v);
						}
					}
				}
			}
		}
		int root = ds.root(0);
		Tree tree = new Tree();
		tree.g = g;
		tree.root = root;
		return tree;
	}

	class Tree {
		ArrayList<Integer>[] g;
		int root;
		int[] h;
		// hash
		long[] rand;
		long p = Long.valueOf(BigInteger.valueOf(500_000_000).nextProbablePrime().toString());

		long hash() {
			h = new int[g.length];
			dfs(root, -1);
			int deg = h[root];
			rand = new long[deg + 1];
			rand[0] = 1;
			for (int i = 1; i < rand.length; ++i) {
				rand[i] = Long.valueOf(BigInteger.valueOf(rand[i - 1]).nextProbablePrime().toString());
			}
			long H = dfs2(root, -1);
			return H;
		}

		long dfs2(int cur, int par) {
			long ret = 1;
			for (int dst : g[cur]) {
				if (dst == par)
					continue;
				ret *= (rand[h[cur]] + dfs2(dst, cur));
				ret %= p;
			}
			return ret;
		}

		void dfs(int cur, int par) {
			for (int dst : g[cur]) {
				if (dst == par) {
					continue;
				}
				dfs(dst, cur);
				h[cur] = Math.max(h[cur], h[dst] + 1);
			}
		}
	}

	class DJSet {
		int n;
		int[] upper;

		public DJSet(int n_) {
			n = n_;
			upper = new int[n];
			Arrays.fill(upper, -1);
		}

		boolean equiv(int x, int y) {
			return root(x) == root(y);
		}

		int root(int x) {
			return upper[x] < 0 ? x : (upper[x] = root(upper[x]));
		}

		void setUnion(int x, int y) {
			x = root(x);
			y = root(y);
			if (x == y) {
				return;
			}
			if (upper[x] < upper[y]) {
				int d = x;
				x = y;
				y = d;
			}
			upper[y] += upper[x];
			upper[x] = y;
		}

		int sz() {
			int ret = 0;
			boolean[] vis = new boolean[n];
			for (int i = 0; i < n; ++i) {
				int v = root(i);
				if (vis[v])
					continue;
				++ret;
				vis[v] = true;
			}
			return ret;
		}

	}

	static void tr(Object... objects) {
		System.out.println(Arrays.deepToString(objects));
	}
}