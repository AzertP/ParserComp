import java.io.*;
import java.util.*;
import java.util.function.Function;

public class Main {

    static int N;

    public static void main(String[] args) {
        FastScanner sc = new FastScanner(System.in);
        N = sc.nextInt();

        System.out.println(solve());
    }

    static int solve() {
        int[] P = getPrimes(N);

        Map<Integer, Integer> cnt = new HashMap<>();
        for (int i = 2; i <= N; i++) {
            merge(cnt, countFactors(i, P));
        }

        int a2 = 0;
        int a4 = 0;
        int a14 = 0;
        int a24 = 0;
        int a74 = 0;
        for (Integer p : cnt.keySet()) {
            int c = cnt.get(p);
            if( c >= 74 ) a74++;
            if( c >= 24 ) a24++;
            if( c >= 14 ) a14++;
            if( c >= 4 ) a4++;
            if( c >= 2 ) a2++;
        }

        // System.out.println(a2 + " " + a4 + " " + a14 + " " + a24 + " " + a74 );

        // 5*5*3
        int ans = 0;
        if( a4 >= 2 && (a2 - 2) >= 1 ) {
            ans += a4 * (a4 - 1) / 2 * (a2 - 2);
        }

        // 25*3
        if( a24 >= 1 && (a2-1) >= 1) {
            ans += a24 * (a2-1);
        }

        // 15*5
        if( a14 >= 1) {
            ans += a14 * (a4-1);
        }

        ans += a74;

        return ans;
    }

    static boolean[] sieveEratos(int n) {
        boolean[] primes = new boolean[n+1];
        Arrays.fill(primes, true);
        primes[0] = false;
        primes[1] = false;

        int root = (int)Math.sqrt(n);
        for (int i = 2; i <= root; i++) {
            if( primes[i] ) {
                for (int d = 2; d*i <= n; d++) {
                    primes[i*d] = false;
                }
            }
        }
        return primes;
    }

    static int[] getPrimes(int n) {
        boolean[] sieve = sieveEratos(n);
        int cnt = 0;
        for (boolean prime : sieve) {
            if( prime ) cnt++;
        }
        int[] primes = new int[cnt];
        int idx = 0;
        for (int num = 2; num < sieve.length; num++) {
            if( sieve[num] ) {
                primes[idx++] = num;
            }
        }
        return primes;
    }

    static void merge(Map<Integer, Integer> map, Map<Integer, Integer> cnt) {
        for (Integer k : cnt.keySet()) {
            int c = cnt.get(k);

            if( map.containsKey(k) ) {
                int c2 = map.get(k);
                map.put(k, c + c2);

            } else {
                map.put(k, 1);
            }
        }
    }

    static Map<Integer, Integer> countFactors(int n, int[] P) {
        Map<Integer, Integer> cnts = new HashMap<>();
        int i = 0;
        for (int p : P) {
            int cnt = 0;
            while( n % p == 0 ) {
                cnt++;
                n /= p;
            }
            if( cnt != 0 ) {
                cnts.put(p, cnt);
            }
        }
        if( n != 1 ) {
            cnts.put(n, 1);
        }
        return cnts;
    }


    @SuppressWarnings("unused")
    static class FastScanner {
        private BufferedReader reader;
        private StringTokenizer tokenizer;

        FastScanner(InputStream in) {
            reader = new BufferedReader(new InputStreamReader(in));
            tokenizer = null;
        }

        String next() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        String nextLine() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    return reader.readLine();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken("\n");
        }

        long nextLong() {
            return Long.parseLong(next());
        }

        int nextInt() {
            return Integer.parseInt(next());
        }

        int[] nextIntArray(int n) {
            int[] a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = nextInt();
            return a;
        }

        int[] nextIntArray(int n, int delta) {
            int[] a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = nextInt() + delta;
            return a;
        }

        long[] nextLongArray(int n) {
            long[] a = new long[n];
            for (int i = 0; i < n; i++)
                a[i] = nextLong();
            return a;
        }
    }

    static <A> void writeLines(A[] as, Function<A, String> f) {
        PrintWriter pw = new PrintWriter(System.out);
        for (A a : as) {
            pw.println(f.apply(a));
        }
        pw.flush();
    }

    static void writeLines(int[] as) {
        PrintWriter pw = new PrintWriter(System.out);
        for (int a : as) pw.println(a);
        pw.flush();
    }

    static void writeLines(long[] as) {
        PrintWriter pw = new PrintWriter(System.out);
        for (long a : as) pw.println(a);
        pw.flush();
    }

    static int max(int... as) {
        int max = Integer.MIN_VALUE;
        for (int a : as) max = Math.max(a, max);
        return max;
    }

    static int min(int... as) {
        int min = Integer.MAX_VALUE;
        for (int a : as) min = Math.min(a, min);
        return min;
    }

    static void debug(Object... args) {
        StringJoiner j = new StringJoiner(" ");
        for (Object arg : args) {
            if (arg instanceof int[]) j.add(Arrays.toString((int[]) arg));
            else if (arg instanceof long[]) j.add(Arrays.toString((long[]) arg));
            else if (arg instanceof double[]) j.add(Arrays.toString((double[]) arg));
            else if (arg instanceof Object[]) j.add(Arrays.toString((Object[]) arg));
            else j.add(arg.toString());
        }
        System.err.println(j.toString());
    }
}
