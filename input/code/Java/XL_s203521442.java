import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.PriorityQueue;
public class Main 
{
    static class Scanner
    {
        BufferedReader br;
        StringTokenizer tk=new StringTokenizer("");
        public Scanner(InputStream is) 
        {
            br=new BufferedReader(new InputStreamReader(is));
        }
        public int nextInt() throws IOException
        {
            if(tk.hasMoreTokens())
                return Integer.parseInt(tk.nextToken());
            tk=new StringTokenizer(br.readLine());
            return nextInt();
        }
        public long nextLong() throws IOException
        {
            if(tk.hasMoreTokens())
                return Long.parseLong(tk.nextToken());
            tk=new StringTokenizer(br.readLine());
            return nextLong();
        }
        public String next() throws IOException
        {
            if(tk.hasMoreTokens())
                return (tk.nextToken());
            tk=new StringTokenizer(br.readLine());
            return next();
        }
        public String nextLine() throws IOException
        {
            tk=new StringTokenizer("");
            return br.readLine();
        }
        public double nextDouble() throws IOException
        {
            if(tk.hasMoreTokens())
                return Double.parseDouble(tk.nextToken());
            tk=new StringTokenizer(br.readLine());
            return nextDouble();
        }
        public char nextChar() throws IOException
        {
            if(tk.hasMoreTokens())
                return (tk.nextToken().charAt(0));
            tk=new StringTokenizer(br.readLine());
            return nextChar();
        }
        public int[] nextIntArray(int n) throws IOException
        {
            int a[]=new int[n];
            for(int i=0;i<n;i++)
                a[i]=nextInt();
            return a;
        }
        public long[] nextLongArray(int n) throws IOException
        {
            long a[]=new long[n];
            for(int i=0;i<n;i++)
                a[i]=nextLong();
            return a;
        }
        public int[] nextIntArrayOneBased(int n) throws IOException
        {
            int a[]=new int[n+1];
            for(int i=1;i<=n;i++)
                a[i]=nextInt();
            return a;
        }
        public long[] nextLongArrayOneBased(int n) throws IOException
        {
            long a[]=new long[n+1];
            for(int i=1;i<=n;i++)
                a[i]=nextLong();
            return a;
        }
    
    
    }
    public static void main(String args[]) throws IOException
    {
        new Thread(null, new Runnable() {
            public void run() {
                try
                {
                    solve();
                }
                catch(Exception e)
                {
                    e.printStackTrace();
                }
            }
        }, "1", 1 << 26).start();
        
    }
    static class Circle{
        long x,y,r;
        Circle(long a,long b,long c){
            x=a;y=b;r=c;
        }
    }
    static class DSU
    {
        int n;
        int p[],r[];
        DSU(int y)
        {
            n=y+10;
            p=new int[n];
            r=new int[n];
            for(int i=0;i<n;i++)
            {
                p[i]=i;
            }
        }
        int find(int x)
        {
            if(p[x]==x)
                return x;
            return p[x]=find(p[x]);
        }
        boolean union(int a,int b)
        {
            a=find(a);
            b=find(b);
            if(a==b)
                return false;
            if(r[a]<r[b])
                p[a]=b;
            else if(r[b]<r[a])
                p[b]=a;
            else
            {
                p[b]=a;
                r[a]++;
            }
            return true;
        }
            
    }
    static class Pair{
        int nd;
        double wt;
        Pair(int a, double b){
            nd=a;
            wt=b;
        }
    }
    static boolean intersect(Circle c1, Circle c2){
        return (c1.x-c2.x)*(c1.x-c2.x)+(c1.y-c2.y)*(c1.y-c2.y)<=(c1.r+c2.r)*(c1.r+c2.r);
    }
    static double distance(Circle c1, Circle c2){
        long d=(c1.x-c2.x)*(c1.x-c2.x)+(c1.y-c2.y)*(c1.y-c2.y);
        double dis=Math.sqrt(d);
        return dis-c1.r-c2.r;
    }
    static void solve() throws IOException
    {
        Scanner in=new Scanner(System.in);
        PrintWriter out=new PrintWriter(System.out);
        ArrayList<Circle> c=new ArrayList<>();
        Circle stc=new Circle(in.nextInt(), in.nextInt(),0);
        Circle endc=new Circle(in.nextInt(), in.nextInt(), 0);
        c.add(stc);
        int n=in.nextInt();
        for(int i=0;i<n;i++){
            c.add(new Circle(in.nextInt(), in.nextInt(), in.nextInt()));
        }
        c.add(endc);     
            
        ArrayList<Pair> g[]=new ArrayList[n+2];
        for(int i=0;i<=n+1;i++)
            g[i]=new ArrayList<>();
        
        for(int i=0;i<=n+1;i++)
            for(int j=i+1;j<=n+1;j++)
                if(true){
                    double dis=Math.max(0,distance(c.get(i), c.get(j)));
                    g[i].add(new Pair(j,dis));
                    g[j].add(new Pair(i,dis));
                }
        
        double dis[]=new double[n+2];
        boolean vis[]=new boolean[n+2];
        PriorityQueue<Pair> pq=new PriorityQueue<>((a,b)->Double.compare(a.wt, b.wt));
        pq.add(new Pair(0,0));
        Arrays.fill(dis,Double.MAX_VALUE);
        dis[0]=0;
        while(!pq.isEmpty()){
            Pair ndx=pq.poll();
            if(vis[ndx.nd])
                continue;
            
            vis[ndx.nd]=true;
            for(Pair edg:g[ndx.nd]){
                double next=ndx.wt+edg.wt;
                if(next<dis[edg.nd]){
                    dis[edg.nd]=next;
                    pq.add(new Pair(edg.nd, next));
                }
            }
            
        }
        out.printf("%.14f",dis[(n+1)]);
        out.close();
        
    }
}
