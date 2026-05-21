import java.util.*;
import java.io.*;
import java.lang.*;
//import java.util.Scanner;
//import java.io.InputStream;
//import java.io.IOException;
//import java.util.NoSuchElementException;
//import java.io.PrintWriter;
//import java.lang.Math;
//import java.io.BufferedReader;
//import java.io.InputStreamReader;
//import java.util.StringTokenizer;

public class Main
{
	public static void main(String[] args)
	{
		StringBuilder sb = new StringBuilder();
		
//		Scanner sc = new Scanner(System.in);
		FastScanner sc = new FastScanner();
		Util39 util = new Util39();
//		int n = sc.nextInt();
//		String s = sc.next();
//		int a[] = new int[3];
//		for (int ii = 0; ii < 3; ii++) a[ii] = sc.nextInt();
		
		
		
		String s = sc.next();
		char[] c = s.toCharArray();
		int N = c.length;
		int A = 0;
		int Z = N - 1;

		for (int i = 0; i < N; i++)
		{
			if (c[i] == 'A')
			{
				A = i;
				break;
			}
		}
		for (int j = Z; j >= 0; j--)
		{
			if (c[j] == 'Z')
			{
				Z = j;
				break;
			}
		}
		sb.append(Z - A + 1);
		
		
//		System.out.println(sb.toString());
		PrintWriter out = new PrintWriter(System.out);
		out.println(sb.toString());
		out.flush();
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
    public boolean hasNext() { while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++; return hasNextByte();}
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
        long nl = nextLong();
        if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE) throw new NumberFormatException();
        return (int) nl;
    }
    public double nextDouble() { return Double.parseDouble(next());}
}

class Util39 {
	public int pow(int a, int b)
	{
		int r = 1;
		for (int i = 0; i < b; i++) r *= a;
		return r;
	}
}
