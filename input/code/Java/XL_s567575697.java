import java.io.*;
import java.time.Year;
import java.util.*;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.IntBinaryOperator;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.*;
import static java.lang.String.format;

public class Main {
    public static void main(String[] args) {
        solve();
    }
    final static long INF = Long.MAX_VALUE>>2;
    final static int MOD = 1_000_000_007;
    final static int[] dx4 = { 0, 1, 0, -1 };
    final static int[] dy4 = { 1, 0, -1, 0 };
    final static int[] dx9 = {-1, -1, -1,  0,  0, 0, 1, 1,1};
    final static int[] dy9 = {-1, 0, 1, -1, 0, 1, -1, 0,1};
    final static Runnable me=()->{};
    public static void solve(){
        //TODO:Solve problem like ***
        Scanner sc=new Scanner();
        int N = sc.nextInt();
        int M = sc.nextInt();
        int R = sc.nextInt();
        int[] r = new int[R];
        for (int i = 0; i < R; i++) {
            r[i] = sc.nextInt()-1;
        }
        Map<Integer, Map<Integer, Integer>> graph = new HashMap<>();
        for (int i = 0; i < M; i++) {
            int a = sc.nextInt()-1;
            int b = sc.nextInt()-1;
            int c = sc.nextInt();
            Map<Integer, Integer> map = graph.getOrDefault(a, new HashMap<>());
            map.put(b, c);
            graph.put(a, map);
            map = graph.getOrDefault(b, new HashMap<>());
            map.put(a, c);
            graph.put(b, map);
        }
        long[][] costs = new long[R][R];
        for (int i = 0; i < R; i++) {
            for (int j = i+1; j < R; j++) {
                //i<j
                long[] cost = new long[N];
                Arrays.fill(cost,Long.MAX_VALUE);
                PriorityQueue<FixedLongPair> pq = new PriorityQueue<>(Comparator.comparingLong(flp -> flp.y));
                pq.add(new FixedLongPair(r[i], 0));
                while (cost[r[j]] == Long.MAX_VALUE) {
                    FixedLongPair flp = pq.remove();
                    if(cost[(int)flp.x]!=Long.MAX_VALUE)continue;
                    cost[(int) flp.x] = flp.y;
                    graph.getOrDefault((int)flp.x,Collections.emptyMap()).forEach((x,y)->{
                        pq.add(new FixedLongPair(x, y + cost[(int) flp.x]));
                    });
                }
//                put(Arrays.toString(cost));
                costs[i][j]=cost[r[j]];
            }
        }
//        putM("costs", costs);
        int[] ans = new int[R];
        for (int i = 0; i < R; i++) {
            ans[i] = i;
        }
        long anss = Long.MAX_VALUE;
        do{
//            put(Arrays.toString(ans));
            long ansss = 0;
            for (int i = 0; i < R-1; i++) {
                int min = min(ans[i], ans[i + 1]);
                int max = max(ans[i], ans[i + 1]);
                ansss += costs[min][max];
            }
            anss = min(anss, ansss);

        }while(Permutation.nextPermutation(ans));
        put(anss);

    }
    final private static class Permutation {
        public static int startIndexOfDescendingOrder(int[] p){
            //return i
            //mean p[i-1]<p[i]&&p[i]>p[i+1]>p[i+2]>...>p[p.length]
            for(int i=p.length-2;i>=0;i--){
                if(p[i]-p[i+1]>=0)continue;
                return i+1;
            }
            return 0;
        }
        public static <T> int startIndexOfDescendingOrder(T[] p,Comparator<? super T> comparator){
            //return i
            //mean p[i-1]<p[i]&&p[i]>p[i+1]>p[i+2]>...>p[p.length]
            for(int i=p.length-2;i>=0;i--){
                if(comparator.compare(p[i],p[i+1])>=0)continue;
                return i+1;
            }
            return 0;
        }
        public static <T> int startIndexOfDescendingOrder(List<T> list,Comparator<? super T> comparator){
            //return i
            //mean p[i-1]<p[i]&&p[i]>p[i+1]>p[i+2]>...>p[p.length]
            for(int i=list.size()-2;i>=0;i--){
                if(comparator.compare(list.get(i),list.get(i+1))>=0)continue;
                return i+1;
            }
            return 0;
        }

        public static boolean nextPermutation(int[] p) {
            int firstIndex=startIndexOfDescendingOrder(p);
            if(firstIndex==0)return false;
            int ng=p.length;
            int ok=firstIndex;
            firstIndex--;
            while(Math.abs(ng-ok)>1){
                int mid=(ng+ok)/2;
                if(p[mid]>p[firstIndex]){
                    ok=mid;
                }else{
                    ng=mid;
                }
            }
            int t = p[firstIndex];
            p[firstIndex] = p[ok];
            p[ok] = t;
            //p[firstIndex+1]以降のものを逆にする
            firstIndex++;
            Arrays.sort(p,firstIndex,p.length);
            return true;
        }
        public static <T> boolean nextPermutation(T[] p,Comparator<? super T> comparator){
            int firstIndex=startIndexOfDescendingOrder(p,comparator);
            if(firstIndex==0)return false;
            int ng=p.length;
            int ok=firstIndex;
            firstIndex--;
            while(Math.abs(ng-ok)>1){
                int mid=(ng+ok)/2;
                if(comparator.compare(p[mid],p[firstIndex])>0){
                    ok=mid;
                }else{
                    ng=mid;
                }
            }
            T t = p[firstIndex];
            p[firstIndex] = p[ok];
            p[ok] = t;
            //p[firstIndex+1]以降のものを逆にする
            //現在降順で昇順に並べる
            //ソートのほうが簡潔に書ける
            firstIndex++;
            Arrays.sort(p,firstIndex,p.length);
            return true;
        }
        public static <T> boolean nextPermutation(List<T> list,Comparator<T> comparator) {
            int firstIndex=startIndexOfDescendingOrder(list,comparator);
            if(firstIndex==0)return false;
            int ng=list.size();
            int ok=firstIndex;
            firstIndex--;
            //MGR式
            while(Math.abs(ng-ok)>1){
                int mid=(ng+ok)/2;
                if(comparator.compare(list.get(mid),list.get(firstIndex))>0){
                    //真に大きいならokの領域
                    ok=mid;
                }else{
                    ng=mid;
                }
            }
            T t = list.get(firstIndex);
            list.set(firstIndex,list.get(ok));
            list.set(ok,t);
            //インデックスがfirstIndex+1以降のものを逆にする
            firstIndex++;
            Collections.reverse(list.subList(firstIndex,list.size()));
            return true;
        }
        public static <T> boolean prevPermutation(List<T> list,Comparator<T> comparator){
            return nextPermutation(list,comparator.reversed());
        }
    }
    static class Accepter<T>{
        private T val;
        private final Predicate p;
        public Accepter(T defaultValue,Predicate p){
            this.val=defaultValue;
            this.p=p;
        }

        /**
         * @return accepted newval?
         */
        public boolean replace(T newVal){
            if(p.test(newVal)){
                this.val=newVal;
                return true;
            }
            return false;
        }
    }
    //runWhenEAで使う
    private static Runnable func(Object... objects){
        try{
            assert false;
            return me;
        }catch(AssertionError e){
            return ()->{put(objects);};
        }
    }
    private static void print(Object... objects){
        if(objects.length==1){
            System.out.print(objects[0]);
        }else{
            System.out.print(Arrays.toString(objects));
        }
    }
    private static void put(Object... objects) {
        print(objects);
        put();
    }
    private static void put(){
        System.out.println();
    }


    private static void runWhenEA(Runnable runnable){
        try{
            assert false;
        }catch(AssertionError e){
            PrintStream ps=System.out;
            PrintStream pse=System.err;
            System.setOut(pse);
            runnable.run();
            System.setOut(ps);
        }
    }
    private static void putM(String name,char[][] mat){
        put("---------------------"+name+"-----------------");
        for (int i = 0; i < mat.length; i++) {
            put(Arrays.toString(mat[i]));
        }
    }
    private static void putM(String name,int[][] mat){
        put("---------------------"+name+"-----------------");
        for (int i = 0; i < mat.length; i++) {
            put(Arrays.toString(mat[i]));
        }
    }
    private static void putM(String name,long[][] mat){
        put("---------------------"+name+"-----------------");
        for (int i = 0; i < mat.length; i++) {
            put(Arrays.toString(mat[i]));
        }
    }
    private static void putM(String name,boolean[][] mat){
        put("---------------------"+name+"-----------------");
        for (int i = 0; i < mat.length; i++) {
            put(Arrays.toString(mat[i]));
        }
    }
    final static private class Scanner {
        private final InputStream in = System.in;
        private final byte[] buffer = new byte[1024];
        private int ptr = 0;
        private int buflen = 0;

        private boolean hasNextByte() {
            if (ptr < buflen) {
                return true;
            } else {
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

        private int readByte() {
            if (hasNextByte())
                return buffer[ptr++];
            else
                return -1;
        }

        private boolean isPrintableChar(int c) {
            return 33 <= c && c <= 126;
        }

        public boolean hasNext() {
            while (hasNextByte() && !isPrintableChar(buffer[ptr]))
                ptr++;
            return hasNextByte();
        }

        public String next() {
            if (!hasNext())
                throw new NoSuchElementException();
            StringBuilder sb = new StringBuilder();
            int b = readByte();
            while (isPrintableChar(b)) {
                sb.appendCodePoint(b);
                b = readByte();
            }
            return sb.toString();
        }

        public long nextLong() {
            if (!hasNext())
                throw new NoSuchElementException();
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
            while (true) {
                if ('0' <= b && b <= '9') {
                    n *= 10;
                    n += b - '0';
                } else if (b == -1 || !isPrintableChar(b)) {
                    return minus ? -n : n;
                } else {
                    throw new NumberFormatException();
                }
                b = readByte();
            }
        }

        public int nextInt() {
            long nl = nextLong();
            if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE)
                throw new NumberFormatException();
            return (int) nl;
        }

        public double nextDouble() {
            return Double.parseDouble(next());
        }
    }
    final static private class FixedIntPair {
        final public int x, y,id;
        static int ids=0;
        final static public FixedIntPair ZEROS=new FixedIntPair(0,0);
        FixedIntPair(int x, int y) {
            this.x = x;
            this.y = y;
            this.id=ids++;
        }
        public static double distance(FixedIntPair fip1,FixedIntPair fip2){
            double x = (double) fip1.x - fip2.x;
            double y = (double) fip1.y - fip2.y;
            return Math.sqrt(x*x+y*y);
        }
        @Override
        public String toString() {
            return format(FixedIntPair.class.getSimpleName()+":(%d,%d)", x, y);
        }
    }
    final static private class FixedLongPair {
        final public long x, y;
        final static public FixedLongPair ZEROS=new FixedLongPair(0,0);
        FixedLongPair(long x, long y) {
            this.x = x;
            this.y = y;
        }
        public static double distance(FixedLongPair flp1,FixedLongPair flp2){
            double x = (double) flp1.x - flp2.x;
            double y = (double) flp1.y - flp2.y;
            return Math.sqrt(x*x+y*y);
        }

        @Override
        public int hashCode() {
            return (int)x+(int)y;
        }

        @Override
        public boolean equals(Object obj) {
            if(obj==null)return false;
            if(!(obj instanceof FixedLongPair)) return false;
            FixedLongPair pair=(FixedLongPair)obj;
            return this.x==pair.x&&this.y==pair.y;
        }

        @Override
        public String toString() {
            return format(FixedLongPair.class.getSimpleName()+":(%d,%d)", x, y);
        }
    }
    final static private class Binary{
        public static String toZeroPadding(int i){
            return format("%"+Integer.toBinaryString(-1).length()+"s",Integer.toBinaryString(i)).replace(' ','0');
        }
        public static String toZeroPadding(long i){
            return format("%"+Long.toBinaryString(-1).length()+"s",Long.toBinaryString(i)).replace(' ','0');
        }
    }
    public static class BinaryIndexedTree {
        public static void main(String... args){
            int a[]={1,3,2,5,7,6,4};
            System.out.println(inv(a));
        }
        final int n;
        final long[] array;
        BinaryIndexedTree(long[] array){
            this.n=array.length;
            this.array=new long[n];
            for(int i=0;i<array.length;i++){
                add(i,array[i]);
            }
        }
        BinaryIndexedTree(int[] array){
            this.n=array.length;
            this.array=new long[n];
            for(int i=0;i<array.length;i++){
                add(i,array[i]);
            }
        }
        public void add(int index,long val){
            for(;index<n;index|=index+1){
                array[index]+=val;
            }
        }
        public long sum(int index){
            long result=0;
            for(;index>=0;index=(index&(index+1))-1){
                result+=array[index];
            }
            return result;
        }

        /**
         *
         * @param array arrayが1～nの順列になっていないならば定義されない
         * @return 転倒数
         */
        public static int inv(int[] array){
            int[] a=new int[array.length];
            BinaryIndexedTree bit=new BinaryIndexedTree(a);
            bit.add(array[0]-1,1);
            int result=0;
            for(int i=1;i<array.length;i++){
                System.out.println(bit);
                result+=i-bit.sum(array[i]-1);
                bit.add(array[i]-1,1);
            }
            return result;
        }

        @Override
        public String toString() {
            long[] array=new long[n];
            for(int i=0;i<n;i++){
                array[i]=sum(i);
            }
            String result="";
            result+="ruiseki:"+ Arrays.toString(array)+"\n";
            array[0]=sum(0);
            for(int i=1;i<n;i++){
                array[i]=sum(i)-sum(i-1);
            }
            result+="source::"+Arrays.toString(array);
            return result;
        }
    }

    final static private class Util {
        static long gcd(long a,long b){
            a= abs(a);
            b= abs(b);
            if(a<b){
                long tmp=a;
                a=b;
                b=tmp;
            }
            while(b!=0){
                long r=a%b;
                a=b;
                b=r;
            }
            return a;
        }
        static int gcd(int a,int b){
            a= abs(a);
            b= abs(b);
            if(a<b){
                int tmp=a;
                a=b;
                b=tmp;
            }
            while(b!=0){
                int r=a%b;
                a=b;
                b=r;
            }
            return a;
        }
        static long lcm(long a,long b){
            long gcd=gcd(a,b);
            long result=b/gcd;
            return a*result;
        }
        static boolean isValidCell(int i,int j,int h,int w){
            return i>=0&&i<h&&j>=0&&j<w;
        }
    }
}































































/*
......................................................................................................................................
.....................................  ________ ____ __________________________________________ .....................................
..................................... /  _____/|    |   \      \__    ___/\_   _____/\______   \.....................................
...................................../   \  ___|    |   /   |   \|    |    |    __)_  |       _/.....................................
.....................................\    \_\  \    |  /    |    \    |    |        \ |    |   \.....................................
..................................... \______  /______/\____|__  /____|   /_______  / |____|_  /.....................................
.....................................        \/                \/                 \/         \/ .....................................
......................................................................................................................................
.............................................................,;'';:...................................................................
........................................................+@@@@@@@@@@@@@@'..............................................................
.....................................................#@@@##############@@@:...........................................................
...................................................@@@####################@@,.........................................................
.................................................@@#########################@@........................................................
...............................................:@############################@@.......................................................
..............................................@@######@@@#';;'#@@@############@@:.....................................................
.............................................@#####@@,````````````,@@###########@:....................................................
............................................@####@;``````````````````+@##########@....................................................
...........................................@###@:``````````````````````#@########@@...................................................
..........................................@####``````````````````````````@########@@..................................................
.........................................###@.````````````````````````````@########@+.................................................
.........................................@#@```````````````````````````````#########@.................................................
........................................@#@`````````````````````````````````########@@................................................
.......................................,@@```````````````````````````````````@#######@:...............................................
.......................................@@`````````````````````````````````````@#######@...............................................
.......................................@:````````````````````#@@'``````````````@######@+..............................................
......................................#@```````````````````@@@@@@@#````````````########@..............................................
......................................@```````````````````@@@@@@@@@@````````````@######@+.............................................
......................................@``````````````````@@@@@@@+   +```````````+#######@.............................................
.....................................;:``````````````````@@@@@@@    @````````````@######@'............................................
.....................................@``````````````````:@@@@@@@    @````````````@#######@............................................
.....................................@```,@@@#``````````;@@@@@@@    @````````````:#######@:...........................................
.....................................@``@@@@@@@@````````.@@@@@@@#  ,#`````````````@#######@...........................................
.....................................@`@@@@@@@+'@````````@@@@@@@@@@@``````````````@#######@...........................................
.....................................@,@@@@@@   ,```:+:``:@@@@@@@@@.``````````````@########@..........................................
.....................................#@@@@@@@  ;@@#;,,,@``:@@@@@@@````````````````#########@..........................................
.....................................+@@@@@@@@',,,,,,,,;,```.'+;``````````````````'########@;.........................................
.....................................'@@@@',,,,,,,,,,,,,@`````````````````````````:#########@.........................................
....................................:@#,,,,,:;;;;;:,,,,,@`````````````````````````.#########@.........................................
.................................:@#@@@@#++';;;;;;;;;;;;@``````````````````````````##########+........................................
...............................#@#+;;;;;;;;;;;;;;;;;;;;':``````````````````````````##########@........................................
....................................,@#@@@@@#+'';;;;;+@#```````````````````````````##########@........................................
.....................................@``````````.,,,.``````````````````````````````############.......................................
.....................................@`````````````````````````````````````````````#######+'+#@.......................................
.....................................@`````````````````````````````````````````````##########'@.......................................
.....................................#`````````````````````````````````````````````############@#.....................................
.....................................:.````````````````````````````````````````````##############@,...................................
......................................+```````````````````````````````````````````.###############@#..................................
......................................@```````````````````````````````````````````.################@@.................................
......................................@```````````````````````````````````````````.###+##############@................................
......................................@```````````````````````````````````````````.###+###############@...............................
......................................',``````````````````````````````````````````.####'##############@@..............................
.......................................@```````````````````````````````````````````#####+##############@:.............................
.......................................@```````````````````````````````````````````#####'###############@.............................
.......................................@```````````````````````````````````````````######'################............................
.......................................#,``````````````````````````````````````````#######'##############@............................
........................................@``````````````````````````````````````````@######++##############+...........................
........................................@``````````````````````````````````````````@#######'##############@...........................
........................................@``````````````````````````````````````````@########'#############@...........................
.......................................@#'`````````````````````````````````````````@#########'##############..........................
.......................................@#@`````````````````````````````````````````+#########+'############@..........................
......................................@##@`````````````````````````````````````````.##########+'###########@..........................
......................................@##@:`````````````````````````````````````````###########+'###########..........................
.....................................:@###@`````````````````````````````````````````@###########+'+#########,.........................
.....................................@####@`````````````````````````````````````````@#############''########..........................
.....................................@####@.````````````````````````````````````````;##############+'######@..........................
.....................................@#####@`````````````````````````````````````````################@@@###+..........................
.....................................@#####@`````````````````````````````````````````@###############@..;;............................
....................................,@#####@.````````````````````````````````````````+################'...............................
....................................:#######@`````````````````````````````````````````################@...............................
....................................:#######@`````````````````````````````````````````@###############@...............................
....................................,@#######,````````````````````````````````````````:###############@...............................
.....................................@######@@`````````````````````````````````````````@##############@...............................
.....................................@######@@`````````````````````````````````````````+##############@...............................
.....................................@#####@,;;`````````````````````````````````````````@#############@...............................
.....................................@####@@..@`````````````````````````````````````````+#############@...............................
.....................................,####@...@``````````````````````````````````````````@############+...............................
......................................@##@.....@`````````````````````````````````````````:###########@,...............................
.......................................@+......@``````````````````````````````````````````@##########@................................
...............................................:#``````````````````````````````````````````##########@................................
................................................@``````````````````````````````````````````+########@,................................
................................................'+``````````````````````````````````````````@#######@.................................
.................................................@```````````````````````````````````````````@#####@:.................................
.................................................'#``````````````````````````````````````````.#####@..................................
..................................................@```````````````````````````````````````````;###@...................................
...................................................@```````````````````````````````````````````+#@'...................................
...................................................'#```````````````````````````````````````````@#....................................
....................................................##`````````````````````````````````````````@#.....................................
.....................................................#@```````````````````````````````````````@+......................................
......................................................:@;```````````````````````````````````;@,.......................................
.......................................................;@@'```````````````````````````````:@@+;.......................................
.......................................................@,,'@@'``````````````````````````@@@,,,@.......................................
......................................................@,,,,,,'@@@@;````````````````.+@@@;,,,,,@.......................................
......................................................#@+@,,,,,,,,+@@@@@@@@@@@@@@@@@;,,,,,'@@@........................................
.........................................................+,,,#',,@@..............@,,,,,,,,@...........................................
..........................................................@@@,#@@,...............:+,,,'@,,@...........................................
..................................................................................@,,,@.##............................................
...................................................................................@;@................................................
....................................................................................:.................................................
......................................................................................................................................
......................................................................................................................................
 */
