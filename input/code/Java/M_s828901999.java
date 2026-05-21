import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

class Main{
    public static void main(String[] args) throws IOException{
        final BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        final String[] in = br.readLine().split(" ");
        final int n = Integer.parseInt(in[0]);
        final int k = Integer.parseInt(in[1]);
        final int[] w = new int[n];
        for(int i = 0; i<n; i++){
            w[i] = Integer.parseInt(br.readLine());
        }
        int l = 0;
        for(final int e:w)
            if(l<e) l = e;
        int r = 0;
        for(int i = k-1; i<n; i++)
            r += w[i];
        r = Math.max(l, r);

        while(l<r){
            final int mid = (l+r)>>>1;
            if(check(w, k, mid)){
                r = mid;
            }else{
                l = mid+1;
            }
        }
        System.out.println(l);
    }
    static private boolean check(int[] w, int k, int p){
        int t = 0;
        int wt = 0;
        for(int i = 0, n = w.length; i<n; i++){
            if(wt+w[i]>p){
                ++t;
                wt = w[i];
                if(t>=k) return false;
            }else{
                wt += w[i];
            }
        }
        return true;
    }
}