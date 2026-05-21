import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.*;
import java.math.BigInteger;
 
public class Main implements Runnable {
	
	static int mod = 1000000007;
	
    public static void main(String[] args) {
    	new Thread(null, new Main(), "", 1024 * 1024 * 1024).start();
    }
    
    public void run() {
       	PrintWriter out = new PrintWriter(System.out);
        FastScanner sc = new FastScanner();

        int n = sc.nextInt();
        int k = sc.nextInt();
        int l = sc.nextInt();
        
        UnionFind car = new UnionFind(n);
        UnionFind train = new UnionFind(n);
        
        for(int i=0;i<k;i++){
        	car.unite(sc.nextInt()-1, sc.nextInt()-1);
        }
        
        for(int i=0;i<l;i++){
        	train.unite(sc.nextInt()-1, sc.nextInt()-1);
        }
        
        HashMap<Pair,Integer> map = new HashMap<>();
        
        for(int i=0;i<n;i++){
        	int p = car.find(i);
        	int q = train.find(i);
        	Pair pair = new Pair(p, q);
        	if(map.containsKey(pair)){
        		map.put(pair, map.get(pair)+1);
        	}
        	else{
        		map.put(pair, 1);
        	}
        }
        
        for(int i=0;i<n-1;i++){
        	int p = car.find(i);
        	int q = train.find(i);
        	Pair pair = new Pair(p, q);
        	out.print(map.get(pair));
        	out.print(" ");
        }
    	int p = car.find(n-1);
    	int q = train.find(n-1);
    	Pair pair = new Pair(p, q);
    	out.println(map.get(pair));
    	
        out.flush();
    }

 
}
class Pair implements Comparable<Pair>{
	int a,b;
	
	public Pair(int a, int b){
		this.a = a;
		this.b = b;
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof Pair){
			Pair p = (Pair) o;
			return a == p.a && b == p.b;
		}
		return super.equals(o);
	}
	
	@Override
	public int compareTo(Pair o){
		if(a!=o.a){
			return Integer.compare(a,o.a);
		}
		return Integer.compare(b, o.b);
	}
	
	@Override
	public int hashCode(){
		return (a<<16)+b;
	}
	
}
class UnionFind {
	int[] par;
	int[] rank;
	int[] peers; //集合の要素数用
	
	public UnionFind(int n){
		par = new int[n];
		rank = new int[n];
		peers = new int[n];
		
		for(int i=0;i<n;i++){
			par[i] = i;
			peers[i] = 1;
		}
	}
	
	//木の根を求める
	int find(int x){
		if(par[x] == x){
			return x;
		}
		else{
			return par[x] = find(par[x]);	//経路圧縮
		}
	}
	
	//xとyの属する集合を併合
	public void unite(int x, int y){
		int px = find(x);
		int py = find(y);
		
		if(px == py){
			return;
		}
		else if(rank[px] < rank[py]){
			peers[py] += peers[px];
			par[px] = py;

		}
		else{
			peers[px] += peers[py];
			par[py] = px;
			if(rank[px]==rank[py]){
				rank[px]++;
			}
		}
		
	}
	
	//xとyが同じ集合に属するか
	public boolean isSame(int x, int y){
		return find(x) == find(y);
	}
	
	//xの仲間の数を求める
	public int setSize(int x){
		return peers[find(x)];
	}

}
 
class FastScanner {
	private final InputStream in = System.in;
	private final byte[] buffer = new byte[1024];
	private int ptr = 0;
	private int buflen = 0;
	private boolean hasNextByte() {
		if (ptr < buflen) {
			return true;
		} else {
			ptr = 0;
			try {
				buflen = in.read(buffer);
			} catch (IOException e) {
				e.printStackTrace();
			}
			if (buflen <= 0) {
				return false;
			}
		}
		return true;
	}
	private int readByte() {
		if (hasNextByte())
			return buffer[ptr++];
		else
			return -1;
	}
	private static boolean isPrintableChar(int c) {
		return 33 <= c && c <= 126;
	}
	public boolean hasNext() {
		while (hasNextByte() && !isPrintableChar(buffer[ptr]))
			ptr++;
		return hasNextByte();
	}
	public String next() {
		if (!hasNext())
			throw new NoSuchElementException();
		StringBuilder sb = new StringBuilder();
		int b = readByte();
		while (isPrintableChar(b)) {
			sb.appendCodePoint(b);
			b = readByte();
		}
		return sb.toString();
	}
	public long nextLong() {
		if (!hasNext())
			throw new NoSuchElementException();
		long n = 0;
		boolean minus = false;
		int b = readByte();
		if (b == '-') {
			minus = true;
			b = readByte();
		}
		if (b < '0' || '9' < b) {
			throw new NumberFormatException();
		}
		while (true) {
			if ('0' <= b && b <= '9') {
				n *= 10;
				n += b - '0';
			} else if (b == -1 || !isPrintableChar(b)) {
				return minus ? -n : n;
			} else {
				throw new NumberFormatException();
			}
			b = readByte();
		}
	}
	public int nextInt() {
		long nl = nextLong();
		if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE)
			throw new NumberFormatException();
		return (int) nl;
	}
	public int[] nextintArray(int n){
		int[] a = new int[n];
		for(int i=0;i<n;i++){
			a[i] = nextInt();
		}
		return a;
	}
	public long[] nextlongArray(int n){
		long[] a = new long[n];
		for(int i=0;i<n;i++){
			a[i] = nextLong();
		}
		return a;
	}
	public Integer[] nextIntegerArray(int n){
		Integer[] a = new Integer[n];
		for(int i=0;i<n;i++){
			a[i] = nextInt();
		}
		return a;
	}
	public int[][] nextintMatrix(int h, int w){
		int[][] mat = new int[h][w];
		for(int i=0;i<h;i++){
			for(int j=0;j<w;j++){
				mat[i][j] = nextInt();
			}
		}
		return mat;
	}
	public double nextDouble() {
		return Double.parseDouble(next());
	}
}