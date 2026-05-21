import java.util.*;
import java.io.*;
import java.math.*;
public class Main {
    static Scanner stdIn = new Scanner(System.in);
    static FastScanner sc = new FastScanner();
    static PrintWriter out = new PrintWriter(System.out);
    public static void main(String[] args) {
    	int d = sc.nextInt();
    	int n = sc.nextInt();
    	int[] t = new int[d];
    	for(int i = 0; i < d; i++) {
    		t[i] = sc.nextInt();
    	}
    	int[] a = new int[n];
    	int[] b = new int[n];
    	int[] c = new int[n];
    	for(int i = 0; i < n; i++) {
    		a[i] = sc.nextInt();
    		b[i] = sc.nextInt();
    		c[i] = sc.nextInt();
    	}
    	
    	int[][] dp = new int[d][101];
    	for(int i = 0; i < d; i++) {
    		Arrays.fill(dp[i], -1);
    	}
    	for(int i = 0; i < n; i++) {
    		if(t[0] >= a[i] && t[0] <= b[i]) {
    			dp[0][c[i]] = 0;
    		}
    	}
    	
    	for(int i = 1; i < d; i++) {
    		for(int j = 0; j < n; j++) {
    			if(t[i] >= a[j] && t[i] <= b[j])
    			for(int k = 0; k < 101; k++) {
    				if(dp[i-1][k] != -1) {
    					dp[i][c[j]] = Math.max(dp[i-1][k] + Math.abs(k - c[j]), dp[i][c[j]]);
    				}
    			}
    		}
    	}

    	
    	int max = 0;
    	for(int i = 0; i < 101; i++) {
    		max = Math.max(dp[d-1][i], max);
    	}
    	out.println(max);
    	out.flush();
    	
    	
    }
    
    static int JOI(char a) {
    	if(a == 'J') return 0;
    	if(a == 'O') return 1;
    	return 2;
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
        }else{
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
    private int readByte() { if (hasNextByte()) return buffer[ptr++]; else return -1;}
    private static boolean isPrintableChar(int c) { return 33 <= c && c <= 126;}
    private void skipUnprintable() { while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++;}
    public boolean hasNext() { skipUnprintable(); return hasNextByte();}
    public String next() {
        if (!hasNext()) throw new NoSuchElementException();
        StringBuilder sb = new StringBuilder();
        int b = readByte();
        while(isPrintableChar(b)) {
            sb.appendCodePoint(b);
            b = readByte();
        }
        return sb.toString();
    }
    public long nextLong() {
        if (!hasNext()) throw new NoSuchElementException();
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
        while(true){
            if ('0' <= b && b <= '9') {
                n *= 10;
                n += b - '0';
            }else if(b == -1 || !isPrintableChar(b)){
                return minus ? -n : n;
            }else{
                throw new NumberFormatException();
            }
            b = readByte();
        }
    }
     
    public int nextInt() {
        return (int)nextLong();
    }
     
    public double nextDouble() {
        return Double.parseDouble(next());
    }
     
 
}