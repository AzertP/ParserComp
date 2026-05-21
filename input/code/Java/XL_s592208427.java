import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        OutputWriter out = new OutputWriter(outputStream);
        Task6665 solver = new Task6665();
        try {
            int testNumber = 1;
            while (true)
                solver.solve(testNumber++, in, out);
        } catch (UnknownError e) {
            out.close();
        }
    }
}

class Task6665 {
    static final int INF = (int) 1e9;
    static final int MOD = 1000003;

    static class Vertex {
        List<Integer> adj = new ArrayList<Integer>();
    }

    int[] hash;
    Vertex[][] g;

    {
        int[] p = new int[9];
        int[] perm = new int[9];
        hash = new int[MOD];
        g = new Vertex[MOD][2];
        for (int i = 0; i < perm.length; i++) {
            perm[i] = i;
        }
        p[8] = 1;
        for (int i = 7; i >= 0; i--) {
            p[i] = p[i + 1] * 10;
        }
        while (true) {
            int s = p[0] * perm[0] + p[1] * perm[1] + p[2] * perm[2] + p[3] * perm[3] + p[4] * perm[4] + p[5] * perm[5] + p[6] * perm[6] + p[7] * perm[7] + p[8] * perm[8];
            int h = calc(s);
            g[h][0] = new Vertex();
            g[h][1] = new Vertex();
            if (!nextPermutation(perm)) {
                break;
            }
        }
        for (int i = 0; i < perm.length; i++) {
            perm[i] = i;
        }
        while (true) {
            int s = p[0] * perm[0] + p[1] * perm[1] + p[2] * perm[2] + p[3] * perm[3] + p[4] * perm[4] + p[5] * perm[5] + p[6] * perm[6] + p[7] * perm[7] + p[8] * perm[8];
            int cur = calc(s);
            for (int i = 0; i < 9; i++) {
                if (perm[i] == 0) {
                    for (int j = -1; j <= 1; j += 2) {
                        int k = i + j;
                        if (k < 0) {
                            k += 9;
                        }
                        if (k >= 9) {
                            k -= 9;
                        }
                        int t = perm[k];
                        perm[k] = perm[i];
                        perm[i] = t;
                        int now = p[0] * perm[0] + p[1] * perm[1] + p[2] * perm[2] + p[3] * perm[3] + p[4] * perm[4] + p[5] * perm[5] + p[6] * perm[6] + p[7] * perm[7] + p[8] * perm[8];
                        g[cur][0].adj.add(calc(now));
                        perm[i] = perm[k];
                        perm[k] = t;
                    }
                    for (int j = -3; j <= 3; j += 6) {
                        int k = i + j;
                        if (k < 0) {
                            k += 9;
                        }
                        if (k >= 9) {
                            k -= 9;
                        }
                        int t = perm[i];
                        perm[i] = perm[k];
                        perm[k] = t;
                        int now = p[0] * perm[0] + p[1] * perm[1] + p[2] * perm[2] + p[3] * perm[3] + p[4] * perm[4] + p[5] * perm[5] + p[6] * perm[6] + p[7] * perm[7] + p[8] * perm[8];
                        g[cur][1].adj.add(calc(now));
                        perm[k] = perm[i];
                        perm[i] = t;
                    }
                    break;
                }
            }
            if (!nextPermutation(perm)) {
                break;
            }
        }
    }

    public void solve(int testNumber, InputReader in, OutputWriter out) {
        int s = 0;
        int t = 0;
        int[] cost = new int[2];
        int[] best = new int[]{0, 1};
        final int[] res = new int[MOD];
        boolean[] mark = new boolean[MOD];
        Queue<Integer> queue = new LinkedList<Integer>();
        for (int i = 0; i < 2; i++) {
            cost[i] = in.readInt();
        }
        if (cost[0] + cost[1] == 0)
            throw new UnknownError();
        for (int i = 0; i < 9; i++) {
            s = s * 10 + in.readInt();
        }
        for (int i = 0; i < 9; i++) {
            t = t * 10 + in.readInt();
        }
        Arrays.fill(res, INF);
        res[calc(s)] = 0;
        mark[calc(s)] = true;
        queue.add(calc(s));
        if (cost[0] > cost[1]) {
            best[0] = 1;
            best[1] = 0;
        }
        int target = calc(t);
        while (!queue.isEmpty()) {
            int x = queue.poll();
            mark[x] = false;
            for (int k = 0; k < 2; k++) {
                int i = best[k];
                for (int j : g[x][i].adj) {
                    int y = res[x] + cost[i];
                    if (res[j] > y && y < res[target]) {
                        res[j] = y;
                        if (!mark[j]) {
                            mark[j] = true;
                            queue.add(j);
                        }
                    }
                }
            }
        }
        out.printLine(res[target]);
    }

    private int calc(int val) {
        int now = val % MOD;
        while (true) {
            if (hash[now] == 0 || hash[now] == val) {
                hash[now] = val;
                break;
            }
            ++now;
        }
        return now;
    }

    private boolean nextPermutation(int[] perm) {
        for (int i = perm.length - 2; i >= 0; i--) {
            if (perm[i] < perm[i + 1]) {
                int tmp = perm[i + 1];
                for (int j = i + 1; j < perm.length; j++) {
                    if (perm[j] > perm[i]) {
                        tmp = Math.min(tmp, perm[j]);
                    }
                }
                for (int j = i + 1; j < perm.length; j++) {
                    if (tmp == perm[j]) {
                        perm[j] = perm[i];
                        perm[i] = tmp;
                        break;
                    }
                }
                Arrays.sort(perm, i + 1, perm.length);
                return true;
            }
        }
        return false;
    }
}

class InputReader {

    private InputStream stream;
    private byte[] buf = new byte[1024];
    private int curChar;
    private int numChars;
    private SpaceCharFilter filter;

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

    public int readInt() {
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
        } while (!isSpaceChar(c));
        return res * sgn;
    }

    public boolean isSpaceChar(int c) {
        if (filter != null)
            return filter.isSpaceChar(c);
        return isWhitespace(c);
    }

    public static boolean isWhitespace(int c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == -1;
    }

    public interface SpaceCharFilter {
        public boolean isSpaceChar(int ch);
    }
}

class OutputWriter {
    private final PrintWriter writer;

    public OutputWriter(OutputStream outputStream) {
        writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(outputStream)));
    }

    public void close() {
        writer.close();
    }

    public void printLine(int i) {
        writer.println(i);
    }
}