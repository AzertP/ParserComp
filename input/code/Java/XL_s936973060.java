import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.BufferedWriter;
import java.io.Writer;
import java.io.OutputStreamWriter;
import java.util.InputMismatchException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 *
 * @author prakharjain
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        OutputWriter out = new OutputWriter(outputStream);
        EOddSubrectangles solver = new EOddSubrectangles();
        solver.solve(1, in, out);
        out.close();
    }

    static class EOddSubrectangles {
        int mod = 998244353;

        long pow(long a, long p, int mod) {
            if (p == 0) {
                return 1;
            }

            long t = pow(a, p / 2, mod);

            if (p % 2 != 0) {
                return (((t * t) % mod) * a) % mod;
            } else {
                return (t * t) % mod;
            }
        }

        public void solve(int testNumber, InputReader in, OutputWriter out) {

            int n = in.nextInt();
            int m = in.nextInt();
            int[][] a = new int[n][m];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    a[i][j] = in.nextInt();
                }
            }

            MatrixRank rank = new MatrixRank();

            int r = rank.matrixRankFF2(a);

            long ans = pow(2, n + m - 1, mod);
            ans -= pow(2, n + m - r - 1, mod);

            ans %= mod;

            if (ans < 0) {
                ans += mod;
            }

            out.println(ans);
        }

        class MatrixRank {
            int matrixRankFF2(int[][] a) {
                int n = a.length;
                int m = a[0].length;

                int rank = 0;

                boolean[] done = new boolean[n];

                for (int i = 0; i < m; i++) {
                    int ind = -1;
                    for (int j = 0; j < n; j++) {
                        if (a[j][i] == 1 && !done[j]) {
                            ind = j;
                            break;
                        }
                    }

                    if (ind == -1)
                        continue;

                    done[ind] = true;
                    rank++;

                    for (int j = ind + 1; j < n; j++) {
                        if (a[j][i] == 1) {
                            for (int k = i; k < m; k++) {
                                a[j][k] ^= a[ind][k];
                            }
                        }
                    }
                }

                return rank;
            }

        }

    }

    static class OutputWriter {
        private final PrintWriter writer;

        public OutputWriter(OutputStream outputStream) {
            writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(outputStream)));
        }

        public OutputWriter(Writer writer) {
            this.writer = new PrintWriter(writer);
        }

        public void close() {
            writer.close();
        }

        public void println(long i) {
            writer.println(i);
        }

    }

    static class InputReader {
        private InputStream stream;
        private byte[] buf = new byte[1024];
        private int curChar;
        private int numChars;
        private InputReader.SpaceCharFilter filter;

        public InputReader(InputStream stream) {
            this.stream = stream;
        }

        public static boolean isWhitespace(int c) {
            return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == -1;
        }

        public int read() {
            if (numChars == -1) {
                throw new InputMismatchException();
            }
            if (curChar >= numChars) {
                curChar = 0;
                try {
                    numChars = stream.read(buf);
                } catch (IOException e) {
                    throw new InputMismatchException();
                }
                if (numChars <= 0) {
                    return -1;
                }
            }
            return buf[curChar++];
        }

        public int nextInt() {
            int c = read();
            while (isSpaceChar(c)) {
                c = read();
            }
            int sgn = 1;
            if (c == '-') {
                sgn = -1;
                c = read();
            }
            int res = 0;
            do {
                if (c < '0' || c > '9') {
                    throw new InputMismatchException();
                }
                res *= 10;
                res += c - '0';
                c = read();
            } while (!isSpaceChar(c));
            return res * sgn;
        }

        public boolean isSpaceChar(int c) {
            if (filter != null) {
                return filter.isSpaceChar(c);
            }
            return isWhitespace(c);
        }

        public interface SpaceCharFilter {
            public boolean isSpaceChar(int ch);

        }

    }
}

