import java.io.*;
import java.util.*;
import static java.lang.System.out;

public class Main {
    static MyReader in = new MyReader();

    public static void main(String[] args) {
        Graph g = new Graph(in);
        long[] d = g.BellmanFord(0);
        out.println(d == null ? "inf" : -d[g.V - 1]);

    }
}

class Graph {
    int V;
    int E;

    int[][] edge;
    ArrayList<int[]>[] T;

    Graph(MyReader in) {
        V = in.i();
        E = in.i();
        edge = new int[E][];
        T = new ArrayList[V];

        Arrays.setAll(T, i -> new ArrayList<>());

        for (int i = 0; i < E; i++) {
            int u = in.i() - 1;
            int v = in.i() - 1;
            int w = in.i();
            edge[i] = new int[]{u, v, w};
            T[u].add(new int[]{v, w});
        }
    }

    long[] BellmanFord(int s) {
        long[] d = new long[V];
        Arrays.fill(d, Long.MAX_VALUE / 2);
        d[s] = 0;

        for (int i = 1; i < V; i++) {
            for (int j = 0; j < E; j++) {
                int u = edge[j][0];
                int v = edge[j][1];
                int w = -edge[j][2];
                if (d[u] + w < d[v]) {
                    d[v] = d[u] + w;
                }
            }
        }

        boolean[] reach = reach(s, V - 1);

        for (int j = 0; j < E; j++) {
            int u = edge[j][0];
            int v = edge[j][1];
            int w = -edge[j][2];
            if (reach[v] && d[u] + w < d[v]) {
                return null;
            }
        }

        return d;
    }

    boolean[] reach(int s, int t) {
        IntQueue q = new IntQueue(V);
        Cons[] parents = new Cons[V];
        boolean[] b = new boolean[V];
        b[s] = b[t] = true;
        q.push(s);
        while (!q.isEmpty()) {
            int u = q.pop();
            for (int[] e : T[u]) {
                int v = e[0];
                if (!b[v]) {
                    q.push(v);
                    b[v] = true;
                }
                parents[v] = new Cons(u, parents[u], parents[v]);
            }
        }

        Arrays.fill(b, false);
        b[t] = true;
        update(parents[t], b);
        return b;
    }

    void update(Cons c, boolean[] reach) {
        if (c != null && !reach[c.N]) {
            update(c.C1, reach);
            update(c.C2, reach);
            reach[c.N] = true;
        }
    }

}

class Cons {
    int N;
    Cons C1, C2;

    Cons(int n, Cons c1, Cons c2) {
        N = n;
        C1 = c1;
        C2 = c2;
    }
}

class IntQueue {
    final int MAX;
    int[] array;
    int head = 0;
    int tail = 0;
    int count = 0;

    IntQueue(int N) {
        MAX = N;
        array = new int[N];
    }

    void push(int n) {
        array[tail] = n;
        if (++tail == MAX) tail = 0;
        count++;
    }

     int pop() {
        int n = array[head];
        if (++head == MAX) head = 0;
        count--;
        return n;
    }

    boolean isEmpty() {
        return count == 0;
    }
}

class MyReader extends BufferedReader {
    char[] cbuf = new char[1024];
    int head = 0;
    int tail = 0;

    MyReader() {
        super(new InputStreamReader(System.in));
    }

    char next() {
        if (head == tail) {
            try {
                tail = super.read(cbuf, 0, cbuf.length);
            } catch (IOException e) {
                e.printStackTrace();
            }
            head = 0;
        }
        return cbuf[head++];
    }

    void back() {
        head--;
    }

    boolean minus() {
        boolean minus;
        while (true) {
            char c = next();
            if (!isDelimiter(c)) {
                if (!(minus = c == '-')) back();
                return minus;
            }
        }
    }

    void skip() {
        while (isDelimiter(next()));
        back();
    }

    char[] s(int N) {
        char[] cbuf = new char[N];
        read(cbuf, 0, N);
        return cbuf;
    }

    public int read(char[] cbuf, int off, int len) {
        skip();
        int i;
        for (i = 0; i < cbuf.length; i++) {
            char c = next();
            if (isDelimiter(c)) {
                break;
            }
            cbuf[i] = c;
        }
        return i;
    }

    boolean isDelimiter(char c) {
        return c == ' ' || c == '\n' || c == '\r';
    }

    int i() {
        boolean minus = minus();
        int n = 0;
        while (true) {
            int k = next() - '0';
            if (k < 0 || 9 < k) break;
            n = 10 * n + k;
        }
        return minus ? -n : n;
    }

    int[] ii(final int N) {
        int[] a = new int[N];
        for (int j = 0; j < a.length; j++) a[j] = i();
        return a;
    }

    long l() {
        boolean minus = minus();
        long n = 0;
        while (true) {
            int k = next() - '0';
            if (k < 0 || 9 < k) break;
            n = 10 * n + k;
        }
        return minus ? -n : n;
    }
}
