import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.InputMismatchException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 *
 * @author KharYusuf
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        DBouquet solver = new DBouquet();
        solver.solve(1, in, out);
        out.close();
    }

    static class DBouquet {
        final long mod = (long) 1e9 + 7;

        public void solve(int testNumber, FastReader s, PrintWriter w) {
            int n = s.nextInt(), a = s.nextInt(), b = s.nextInt();
            long ans = func.power(2, n, mod) - 1;
            ans -= combo(n, b);
            ans -= combo(n, a);
            ans += mod * 2;
            ans %= mod;
            w.println(ans);
        }

        long combo(int n, int m) {
            int n1 = m, n2 = n - m;
            if (n1 < n2) {
                int t = n1;
                n1 = n2;
                n2 = t;
            }
            long ans = 1, div = 1;
            for (int i = n; i > n1; i--) {
                ans *= i;
                ans %= mod;
            }
            for (int i = 2; i <= n2; i++) {
                div *= i;
                div %= mod;
            }
            ans *= func.modinv(div, mod);
            return ans %= mod;
        }

    }

    static class FastReader {
        private InputStream stream;
        private byte[] buf = new byte[1024];
        private int curChar;
        private int numChars;
        private FastReader.SpaceCharFilter filter;

        public FastReader(InputStream stream) {
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

        public boolean isSpaceChar(int c) {

            if (filter != null)
                return filter.isSpaceChar(c);

            return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == -1;
        }

        public interface SpaceCharFilter {
            public boolean isSpaceChar(int ch);

        }

    }

    static class func {
        public static long power(long x, long y, long p) {
            long res = 1;
            x %= p;
            while (y > 0) {
                if ((y & 1) == 1) {
                    res *= x;
                    res %= p;
                }
                y >>= 1;
                x *= x;
                x %= p;
            }
            return res;
        }

        public static long modinv(long a, long m) {
            long m0 = m, y = 0, x = 1, q, t;
            if (m == 1) return 0;
            while (a > 1) {
                q = a / m;
                t = m;
                m = a % m;
                a = t;
                t = y;
                y = x - q * y;
                x = t;
            }
            if (x < 0) x += m0;
            return x;
        }

    }
}

