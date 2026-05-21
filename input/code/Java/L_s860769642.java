
import java.awt.geom.Point2D;
import java.io.*;
import java.math.BigInteger;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.Stack;
import java.util.TreeMap;



public class Main {

	static PrintWriter out = new PrintWriter(System.out);
	static FastScanner sc = new FastScanner();
	static Scanner stdIn = new Scanner(System.in);
	static char[][] c= new char[][] {{'*','*','*','*','*',' ',' ',' ',' ',' '},
									{' ',' ',' ',' ',' ','*','*','*','*','*'},
									{'=','=','=','=','=','=','=','=','=','='},
									{' ','*','*','*','*',' ','*','*','*','*'},
									{'*',' ','*','*','*','*',' ','*','*','*'},
									{'*','*',' ','*','*','*','*',' ','*','*'},
									{'*','*','*',' ','*','*','*','*',' ','*'},
									{'*','*','*','*',' ','*','*','*','*',' '}};
	public static void main(String[] args) {
		boolean FC = false;
		while(sc.hasNext()) {
			if(!FC) {
				FC = true;
			}
			else {
				out.println();
			}
			
			String in = sc.next();
			String tmp = "";
			for(int i = 0; i < 5 - in.length(); i++) {
				tmp += "0";
			}
			tmp += in;
			for(int i = 0; i < 8; i++) {
				for(int j = 0; j < 5; j++) {
					out.print(c[i][tmp.charAt(j)-'0']);
				}
				out.println();
			}
		}
		out.flush();
	}
	
	public static long s(int a, int b) {
		long x = 1;
		long c1 = 0;
		while(true) {
			c1++;
			x = (a*x)%b;
			if(x == 1) {
				break;
			}
		}
		return c1;
	}
	
	public static long gcd(long a,long b) {
		if(b == 0) return a;
		return gcd(b, a%b);
	}
	
	
	
}
//------------------------------//
//-----------//
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
    	 if (!hasNext()) throw new NoSuchElementException();
         int n = 0;
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
    
    public double nextDouble() {
    	return Double.parseDouble(next());
    }
    

}