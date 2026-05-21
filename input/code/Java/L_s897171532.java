import java.util.*;
class Main{
    public static void main(String[] args){
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        List<Place> list = new ArrayList<>();
        for(int i = 0; i < n; i++){
            double x = scanner.nextDouble();
            double y = scanner.nextDouble();
            list.add(new Place(x,y));
        }
        int[] C = new int[n];
        for (int i = 0; i < n; i++) {
            C[i] = i;
        }
        double ans = 0;
        double count = 0;
        do{
            for(int i = 0; i < n-1; i++){
                double tmp = getDistance(list.get(C[i]), list.get(C[i+1]));
                //System.out.println(tmp);
                ans += tmp;
            }
            count++;
        }while(nextPermutation(C));
        System.out.println(ans/count);
    }

    public static double getDistance(Place a, Place b){
        return Math.sqrt(Math.pow((a.x-b.x),2) + Math.pow((a.y-b.y),2));
    }

    static class Place implements Comparable<Place>{
        double x, y;
        public Place(double x, double y){
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object v){
            boolean retVal = false;
            if(v instanceof Place){
                Place ptr = (Place) v;
                retVal = (ptr.x == this.x) && (ptr.y == this.y);
            }
            return retVal;
        }

        @Override
        public int compareTo(Place r){
            return (int)r.x - (int)this.x;
        }
    }

    //【参考URL】
    // http://d.hatena.ne.jp/tomerun/20081203/1228321480
    public static boolean nextPermutation(int[] a) {
    for (int i = a.length - 1; 0 < i; i--) {
      if (a[i - 1] < a[i]) {
        int index = find(a[i - 1], a, i, a.length - 1);
        int temp = a[index];
        a[index] = a[i - 1];
        a[i - 1] = temp;
        Arrays.sort(a, i, a.length);
        return true;
      }
    }
    return false;
  }

    // destより大きい要素の中で最小のもののインデックスを二分探索で探す
    private static int find(int dest, int[] a, int s, int e) {
        if (s == e) {
        return s;
        }
        int m = (s + e + 1) / 2;
        return a[m] <= dest ? find(dest, a, s, m - 1) : find(dest, a, m, e);
    }
}

