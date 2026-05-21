import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.stream.IntStream;
import java.util.Arrays;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.OptionalInt;
import java.io.UncheckedIOException;
import java.util.stream.Stream;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.util.NoSuchElementException;
import java.io.InputStream;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 *
 * @author mikit
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        LightScanner in = new LightScanner(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        CTakahashisInformation solver = new CTakahashisInformation();
        solver.solve(1, in, out);
        out.close();
    }

    static class CTakahashisInformation {
        private static final String YES = "Yes";
        private static final String NO = "No";

        public void solve(int testNumber, LightScanner in, PrintWriter out) {
            int[][] c = in.ints(3, 3);
            for (int i = 0; i < 3; i++) {
                int min = IntMath.min(c[i]);
                for (int j = 0; j < 3; j++) {
                    c[i][j] -= min;
                }
            }
            for (int i = 0; i < 3; i++) {
                if (c[0][i] != c[1][i] || c[1][i] != c[2][i]) {
                    out.println(NO);
                    return;
                }
            }
            out.println(YES);
        }

    }

    static class LightScanner {
        private BufferedReader reader = null;
        private StringTokenizer tokenizer = null;

        public LightScanner(InputStream in) {
            reader = new BufferedReader(new InputStreamReader(in));
        }

        public String string() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int ints() {
            return Integer.parseInt(string());
        }

        public int[] ints(int length) {
            return IntStream.range(0, length).map(x -> ints()).toArray();
        }

        public int[][] ints(int height, int width) {
            return IntStream.range(0, height).mapToObj(x -> ints(width)).toArray(int[][]::new);
        }

    }

    static final class IntMath {
        private IntMath() {
        }

        public static int min(int... v) {
            return Arrays.stream(v).min().orElseThrow(NoSuchElementException::new);
        }

    }
}

