import javax.print.DocFlavor;
import java.awt.*;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.math.BigInteger;
import java.util.*;
import java.util.List;

public class Main {

    static String br = System.getProperty("line.separator");
    static Scanner sc = new Scanner(System.in);
    static PrintWriter out = new PrintWriter(System.out);

    private static int Q;
    private static int N;
    private static int M;
    private static int[] a;
    private static int[] b;
    private static int[] c;
    private static int[] d;

    public static void main(String[] args) {
        int N = nextInt();
        int M = nextInt();
        int[] H = new int[N];
        Arrays.setAll(H, i->nextInt());
        HashMap<Integer, List<Integer>> map = new HashMap<>();
        HashMap<Integer, List<Integer>> map2 = new HashMap<>();
        for(int i=0; i<M; i++){
            int A = nextInt();
            int B = nextInt();
            if(map.containsKey((Integer)A)){
                List<Integer> list = map.get((Integer)A);
                list.add(B);
                map.put(A, list);
            }else{
                List<Integer> list = new ArrayList<>();
                list.add(B);
                map.put(A, list);
            }
        }

        boolean[] isOK = new boolean[N];
        Arrays.fill(isOK, true);
        for(int key : map.keySet()){
            for(Integer value : map.get(key)){
                //println(key + "," + value + "," + (H[key-1] <= H[value-1]));
                if(H[key-1] <= H[value-1]){
                    isOK[key-1] = false;
                }
                if(H[value-1] <= H[key-1]){
                    isOK[value-1] = false;
                }
            }
        }
        int count = 0;
        for(boolean ok : isOK){
            if(ok) count++;
        }

        println(count);

    }

    static int nextInt() {
        return Integer.parseInt(next());
    }

    static Long nextLong() {
        return Long.parseLong(next());
    }

    static String next() {
        return sc.next();
    }

    static String nextLine(){
        return sc.nextLine();
    }

    static void println(Object o) {
        out.println(o);
        out.flush();
    }

    static void print(Object o) {
        PrintWriter out = new PrintWriter(System.out);
        out.print(o);
        out.flush();
    }
}
