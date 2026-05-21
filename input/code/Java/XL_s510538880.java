import java.util.*;
import java.io.*;

public class Main {
	static final int mod = (int) 1e9 + 7;
	static final int DX[] = { -1, 0, 1, 0 }, DY[] = { 0, -1, 0, 1 };
	static final int[] DX8 = { -1, -1, -1, 0, 0, 1, 1, 1 }, DY8 = { -1, 0, 1, -1, 1, -1, 0, 1 };
	static final int INF = Integer.MAX_VALUE / 3;
	static final long LINF = Long.MAX_VALUE / 3;
	static final String nextLine = "\n";

	static int n;
	static List<Integer> graph[];
	static long dp1[] = new long[220000];//根から葉方向で、頂点i以下の部分木の採番の場合の数
	static long dp2[] = new long[220000];//葉から根方向
	static int size1[] = new int[220000];//根から葉方向で、頂点i以下の部分木の大きさ
	static int size2[] = new int[220000];//葉から根方向(size2 == n - size1 + 1)
	static long ans[] = new long[220000];
	public static void main(String[] args) {
		FastScanner fs = new FastScanner(System.in);
		n = fs.nextInt();
		graph = new ArrayList[n];
		for(int i=0;i<n;i++)graph[i] = new ArrayList<>();
		for(int i=0;i<n-1;i++) {
			int a = fs.nextInt()-1, b = fs.nextInt()-1;
			graph[a].add(b);
			graph[b].add(a);
		}
		preFactrials(n+20);
		dfs1(0,-1);
		dfs2(0,-1);
		dfs3(0,-1);//ansの作成(親はdp2,size2を使うため、親が誰か知りたい。よってdfsで作る)
		for(int i=0;i<n;i++) {
			System.out.println(ans[i]);
		}
	}
	
	//根から葉方向の場合の数(dp1)と、部分木の大きさ(size1&size2)を求める
	static void dfs1(int v, int p) {
		dp1[v] = 1;
		size1[v] = 1;//自分を含めるために、1スタート
		for(int to:graph[v]) {
			if(to==p)continue;
			dfs1(to, v);
			size1[v] += size1[to];
			dp1[v] = mul(dp1[v],dp1[to]);
			dp1[v] = mul(dp1[v],factorialDivs[size1[to]]);
		}
		size2[v] = n - size1[v];
		dp1[v] = mul(dp1[v], factorials[size1[v]-1]);
	}
	
	//葉から根
	static void dfs2(int v, int p) {
		if(p==-1)dp2[v] = 1;
		long all_product = 1;
		all_product = mul(all_product, dp2[v]);
		all_product = mul(all_product, factorialDivs[size2[v]]);
		for(int to:graph[v]) {
			if(to == p) continue;
			all_product = mul(all_product, dp1[to]);
			all_product = mul(all_product, factorialDivs[size1[to]]);				
		}
		for(int to:graph[v]) {
			if(to == p)continue;
			dp2[to] = div(all_product, dp1[to]);
			dp2[to] = div(dp2[to],factorialDivs[size1[to]]);
			dp2[to] = mul(dp2[to], factorials[size2[to]-1]);
			dfs2(to, v);
		}
	}
	
	static void dfs3(int v, int p) {
		long res = 1;
		res = mul(res, dp2[v]);
		res = mul(res, factorialDivs[size2[v]]);
		for(int to:graph[v]) {
			if(to == p) continue;
			res = mul(res, dp1[to]);
			res = mul(res, factorialDivs[size1[to]]);				
			dfs3(to, v);
		}
		res = mul(res, factorials[n-1]);
		ans[v] = res;
		
	}

	//Fast Fermat Combination
	static long factorials[];
	static long factorialDivs[];

	static void preFactrials(int n) {
		factorials = new long[n + 1];
		factorialDivs = new long[n + 1];
		factorials[0] = 1;
		for (int i = 0; i < n; i++) {
			factorials[i + 1] = mul(i + 1, factorials[i]);
		}
		factorialDivs[n] = div(1, factorials[n]);
		for (int i = n - 1; i >= 0; i--) {
			factorialDivs[i] = mul(factorialDivs[i + 1], i + 1);
		}
	}

	static long fastFermatComb(int a, int b) {
		if (factorials.length == 0)
			System.err.println("error : factorials has not been culculated!! do [pre_factorial] method");
		long af = factorials[a];
		long bf = factorialDivs[b];
		long abf = factorialDivs[a - b];
		long res = mul(mul(af, bf), abf);
		return res;
	}
	
	
	//MOD culculations
	static long plus(long x, long y) {
		long res = (x + y) % mod;
		return res < 0 ? res + mod : res;
	}

	static long sub(long x, long y) {
		long res = (x - y) % mod;
		return res < 0 ? res + mod : res;
	}

	static long mul(long x, long y) {
		long res = (x * y) % mod;
		return res < 0 ? res + mod : res;
	}

	static long div(long x, long y) {
		long res = x * pow(y, mod - 2) % mod;
		return res < 0 ? res + mod : res;
	}

	static long pow(long x, long y) {
		if (y < 0)
			return 0;
		if (y == 0)
			return 1;
		if (y % 2 == 1)
			return (x * pow(x, y - 1)) % mod;
		long root = pow(x, y / 2);
		return root * root % mod;
	}
}

//高速なScanner
class FastScanner {
	private BufferedReader reader = null;
	private StringTokenizer tokenizer = null;

	public FastScanner(InputStream in) {
		reader = new BufferedReader(new InputStreamReader(in));
		tokenizer = null;
	}

	public String next() {
		if (tokenizer == null || !tokenizer.hasMoreTokens()) {
			try {
				tokenizer = new StringTokenizer(reader.readLine());
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken();
	}

	public String nextLine() {
		if (tokenizer == null || !tokenizer.hasMoreTokens()) {
			try {
				return reader.readLine();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken("\n");
	}

	public long nextLong() {
		return Long.parseLong(next());
	}

	public int nextInt() {
		return Integer.parseInt(next());
	}

	public double nextDouble() {
		return Double.parseDouble(next());
	}

	public int[] nextIntArray(int n) {
		int[] a = new int[n];
		for (int i = 0; i < n; i++)
			a[i] = nextInt();
		return a;
	}

	public long[] nextLongArray(int n) {
		long[] a = new long[n];
		for (int i = 0; i < n; i++)
			a[i] = nextLong();
		return a;
	}
}