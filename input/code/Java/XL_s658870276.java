import java.io.*;
import java.math.*;
import java.util.*;
import java.lang.*;

class Main implements Runnable {
    static class InputReader {
        private InputStream stream;
        private byte[] buf = new byte[1024];
        private int curChar;
        private int numChars;
        private SpaceCharFilter filter;
        private BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        public InputReader(InputStream stream) {
            this.stream = stream;
        }

        public int read() {
            if (numChars == -1)
                throw new InputMismatchException();

            if (curChar >= numChars) {
                curChar = 0;
                try {
                    numChars = stream.read(buf);
                } catch (IOException e) {
                    throw new InputMismatchException();
                }

                if (numChars <= 0)
                    return -1;
            }
            return buf[curChar++];
        }

        public String nextLine() {
            String str = "";
            try {
                str = br.readLine();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return str;
        }

        public int nextInt() {
            int c = read();

            while (isSpaceChar(c))
                c = read();

            int sgn = 1;

            if (c == '-') {
                sgn = -1;
                c = read();
            }

            int res = 0;
            do {
                if (c < '0' || c > '9')
                    throw new InputMismatchException();
                res *= 10;
                res += c - '0';
                c = read();
            }
            while (!isSpaceChar(c));

            return res * sgn;
        }

        public long nextLong() {
            int c = read();
            while (isSpaceChar(c))
                c = read();
            int sgn = 1;
            if (c == '-') {
                sgn = -1;
                c = read();
            }
            long res = 0;

            do {
                if (c < '0' || c > '9')
                    throw new InputMismatchException();
                res *= 10;
                res += c - '0';
                c = read();
            }
            while (!isSpaceChar(c));
            return res * sgn;
        }

        public double nextDouble() {
            int c = read();
            while (isSpaceChar(c))
                c = read();
            int sgn = 1;
            if (c == '-') {
                sgn = -1;
                c = read();
            }
            double res = 0;
            while (!isSpaceChar(c) && c != '.') {
                if (c == 'e' || c == 'E')
                    return res * Math.pow(10, nextInt());
                if (c < '0' || c > '9')
                    throw new InputMismatchException();
                res *= 10;
                res += c - '0';
                c = read();
            }
            if (c == '.') {
                c = read();
                double m = 1;
                while (!isSpaceChar(c)) {
                    if (c == 'e' || c == 'E')
                        return res * Math.pow(10, nextInt());
                    if (c < '0' || c > '9')
                        throw new InputMismatchException();
                    m /= 10;
                    res += (c - '0') * m;
                    c = read();
                }
            }
            return res * sgn;
        }

        public String readString() {
            int c = read();
            while (isSpaceChar(c))
                c = read();
            StringBuilder res = new StringBuilder();
            do {
                res.appendCodePoint(c);
                c = read();
            }
            while (!isSpaceChar(c));

            return res.toString();
        }

        public boolean isSpaceChar(int c) {
            if (filter != null)
                return filter.isSpaceChar(c);
            return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == -1;
        }

        public String next() {
            return readString();
        }

        public interface SpaceCharFilter {
            public boolean isSpaceChar(int ch);
        }
    }

    public static void main(String args[]) throws Exception {
        new Thread(null, new Main(), "Main", 1 << 26).start();
    }

    static long gcd(long a, long b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    static long modPow(long a, long p, long m) {
        if (a == 1) return 1;
        long ans = 1;
        while (p > 0) {
            if (p % 2 == 1) ans = (ans * a) % m;
            a = (a * a) % m;
            p >>= 1;
        }
        return ans;
    }

    static long modInv(long a, long m) {
        return modPow(a, m - 2, m);
    }


    static class Radix {


        static int getMax(int arr[], int n)
        {
            int mx = arr[0];
            for (int i = 1; i < n; i++)
                if (arr[i] > mx)
                    mx = arr[i];
            return mx;
        }

        static void countSort(int arr[], int n, int exp)
        {
            int output[] = new int[n];
            int i;
            int count[] = new int[10];
            Arrays.fill(count,0);

            for (i = 0; i < n; i++)
                count[ (arr[i]/exp)%10 ]++;

            for (i = 1; i < 10; i++)
                count[i] += count[i - 1];

            for (i = n - 1; i >= 0; i--)
            {
                output[count[ (arr[i]/exp)%10 ] - 1] = arr[i];
                count[ (arr[i]/exp)%10 ]--;
            }

            for (i = 0; i < n; i++)
                arr[i] = output[i];
        }

        static void radixsort(int arr[], int n)
        {
            int m = getMax(arr, n);

            for (int exp = 1; m/exp > 0; exp *= 10)
                countSort(arr, n, exp);
        }

    }

    class DSU{
        int par[],size[],connectedComponent;
        DSU(int n){
            par=new int[n];
            size=new int[n];
            for (int i = 0; i <n ; i++) {
                size[i]=1;
                par[i]=i;
            }
            connectedComponent=n;
        }
        int root(int n){
            if(n==par[n])return n;
            return par[n]=root(par[n]);
        }
        boolean union(int u,int v){
            return f_union(root(u),root(v));
        }
        boolean f_union(int u,int v){
            if(u==v)return false;
            if(size[u]>size[v]){
                size[u]+=size[v];
                par[v]=u;
            }
            else{
                size[v]+=size[u];
                par[u]=v;
            }
            connectedComponent--;
            return true;
        }
        DSU clone_DSU(){
            DSU t=new DSU(par.length);
            t.connectedComponent=connectedComponent;
            for (int i = 0; i <par.length ; i++) {
                t.par[i]=par[i];
                t.size[i]=size[i];
            }
            return t;
        }
    }



    public void run() {
        InputReader sc = new InputReader(System.in);
        //Scanner sc=new Scanner(System.in);
        //Random sc=new Random();
        PrintWriter out = new PrintWriter(System.out);


        int a[]=new int[3];
        for (int i = 0; i <3 ; i++) {
            a[i]=sc.nextInt();
        }
        Arrays.sort(a);
        out.println(a[0]+a[1]);


        out.close();
    }
    
}