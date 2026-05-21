import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.io.IOException;
import java.util.ArrayList;
import java.io.UncheckedIOException;
import java.util.List;
import java.io.Closeable;
import java.io.Writer;
import java.io.OutputStreamWriter;
import java.util.Comparator;
import java.io.InputStream;

/**
 * Built using CHelper plug-in Actual solution is at the top
 */
public class Main {
    public static void main(String[] args) throws Exception {
        Thread thread = new Thread(null, new TaskAdapter(), "", 1 << 27);
        thread.start();
        thread.join();
    }

    static class TaskAdapter implements Runnable {
        @Override
        public void run() {
            InputStream inputStream = System.in;
            OutputStream outputStream = System.out;
            FastInput in = new FastInput(inputStream);
            FastOutput out = new FastOutput(outputStream);
            TaskF solver = new TaskF();
            solver.solve(1, in, out);
            out.close();
        }
    }
    static class TaskF {
        public void solve(int testNumber, FastInput in, FastOutput out) {
            int n = in.readInt();
            int[] scores = new int[n];
            for (int i = 0; i < n; i++) {
                scores[i] = in.readInt();
            }

            IntList factors = new IntList(1000);

            List<Query>[] leftQueries = new List[n];
            List<Query>[] rightQueries = new List[n];
            List<Result> results = new ArrayList<>(3000000);
            for (int i = 0; i < n; i++) {
                leftQueries[i] = new ArrayList<>();
                rightQueries[i] = new ArrayList<>();
            }

            for (int a = 1; a < n; a++) {
                int remain = n - 1 - a;
                factors.clear();
                for (int t = 1; t * t <= remain; t++) {
                    if (remain % t != 0) {
                        continue;
                    }
                    factors.add(t);
                    if (t * t != remain) {
                        factors.add(remain / t);
                    }
                }

                if (a == n - 1) {
                    factors.add(1);
                }
                for (int i = 0; i < factors.size(); i++) {
                    int t = factors.get(i);
                    int b = a - t;
                    if (b >= a || b < 0) {
                        continue;
                    }
                    int k = remain / t;
                    if (b % (a - b) == 0) {
                        int meet = b / (a - b) + 1;
                        if (meet <= k) {
                            continue;
                        }
                    }

                    Query left = new Query();
                    left.step = a - b;
                    left.until = k * left.step;
                    leftQueries[left.step].add(left);

                    Query right = new Query();
                    right.step = a - b;
                    right.until = a;
                    rightQueries[right.step].add(right);

                    Result result = new Result();
                    result.left = left;
                    result.right = right;
                    results.add(result);
                }
            }

            for (int i = 0; i < n; i++) {
                leftQueries[i].sort(Query.sortByUntil);
                rightQueries[i].sort(Query.reverseSortByUntil);
            }

            for (int i = 1; i < n; i++) {
                int scan = 0;
                int m = leftQueries[i].size();
                long sum = 0;
                for (int j = i; scan < m; j += i) {
                    while (scan < m && leftQueries[i].get(scan).until < j) {
                        leftQueries[i].get(scan).ans = sum;
                        scan++;
                    }
                    if (j >= n) {
                        break;
                    }
                    sum += scores[j];
                }
            }

            for (int i = 1; i < n; i++) {
                int scan = 0;
                int m = rightQueries[i].size();
                long sum = 0;
                for (int j = n - 1; scan < m; j -= i) {
                    while (scan < m && rightQueries[i].get(scan).until > j) {
                        rightQueries[i].get(scan).ans = sum;
                        scan++;
                    }
                    if (j <= 0) {
                        break;
                    }
                    sum += scores[j];
                }
            }

            long ans = -(long) 1e18;
            for (Result r : results) {
                long sum = r.left.ans + r.right.ans;
                ans = Math.max(ans, sum);
            }

            out.println(ans);
        }

    }
    static class FastOutput implements AutoCloseable, Closeable {
        private StringBuilder cache = new StringBuilder(10 << 20);
        private final Writer os;

        public FastOutput(Writer os) {
            this.os = os;
        }

        public FastOutput(OutputStream os) {
            this(new OutputStreamWriter(os));
        }

        public FastOutput println(long c) {
            cache.append(c).append('\n');
            return this;
        }

        public FastOutput flush() {
            try {
                os.append(cache);
                os.flush();
                cache.setLength(0);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            return this;
        }

        public void close() {
            flush();
            try {
                os.close();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public String toString() {
            return cache.toString();
        }

    }
    static class IntList {
        private int size;
        private int cap;
        private int[] data;
        private static final int[] EMPTY = new int[0];

        public IntList(int cap) {
            this.cap = cap;
            if (cap == 0) {
                data = EMPTY;
            } else {
                data = new int[cap];
            }
        }

        public IntList(IntList list) {
            this.size = list.size;
            this.cap = list.cap;
            this.data = Arrays.copyOf(list.data, size);
        }

        public IntList() {
            this(0);
        }

        private void ensureSpace(int need) {
            int req = size + need;
            if (req > cap) {
                while (cap < req) {
                    cap = Math.max(cap + 10, 2 * cap);
                }
                data = Arrays.copyOf(data, cap);
            }
        }

        private void checkRange(int i) {
            if (i < 0 || i >= size) {
                throw new ArrayIndexOutOfBoundsException();
            }
        }

        public int get(int i) {
            checkRange(i);
            return data[i];
        }

        public void add(int x) {
            ensureSpace(1);
            data[size++] = x;
        }

        public int size() {
            return size;
        }

        public int[] toArray() {
            return Arrays.copyOf(data, size);
        }

        public void clear() {
            size = 0;
        }

        public String toString() {
            return Arrays.toString(toArray());
        }

    }
    static class Result {
        Query left;
        Query right;

    }
    static class FastInput {
        private final InputStream is;
        private byte[] buf = new byte[1 << 13];
        private int bufLen;
        private int bufOffset;
        private int next;

        public FastInput(InputStream is) {
            this.is = is;
        }

        private int read() {
            while (bufLen == bufOffset) {
                bufOffset = 0;
                try {
                    bufLen = is.read(buf);
                } catch (IOException e) {
                    bufLen = -1;
                }
                if (bufLen == -1) {
                    return -1;
                }
            }
            return buf[bufOffset++];
        }

        public void skipBlank() {
            while (next >= 0 && next <= 32) {
                next = read();
            }
        }

        public int readInt() {
            int sign = 1;

            skipBlank();
            if (next == '+' || next == '-') {
                sign = next == '+' ? 1 : -1;
                next = read();
            }

            int val = 0;
            if (sign == 1) {
                while (next >= '0' && next <= '9') {
                    val = val * 10 + next - '0';
                    next = read();
                }
            } else {
                while (next >= '0' && next <= '9') {
                    val = val * 10 - next + '0';
                    next = read();
                }
            }

            return val;
        }

    }
    static class Query {
        static Comparator<Query> sortByUntil = (a, b) -> Integer.compare(a.until, b.until);
        static Comparator<Query> reverseSortByUntil = (a, b) -> -Integer.compare(a.until, b.until);
        int step;
        int until;
        long ans;

    }
}

