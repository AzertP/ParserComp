import java.io.*;import java.util.*;
import java.math.*;

public class Main{
	static ArrayList<Integer> ar[];
	static int n;
	static int v[];
	static int cc=0;
	static int arr[];
	static int var=0,zz=-1,zz2=0;
	static int rev[];
	static HashMap<Integer,Integer> hm=new HashMap<>();
	static int max=Integer.MAX_VALUE,min=Integer.MIN_VALUE;
	static BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
	static StringTokenizer st;
    static int max(int a,int b)
    {
    	return Math.max(a, b);
    }
    static int min(int a,int b)
    {
    	return Math.min(a, b);
    }
    static int i()throws IOException
    {
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	return Integer.parseInt(st.nextToken());
    }
    static long l()throws IOException
    {
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	return Long.parseLong(st.nextToken());
    }
    static String s()throws IOException
    {
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	return st.nextToken();
    }
    static double d()throws IOException
    {
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	return Double.parseDouble(st.nextToken());
    }
    static void p(String p)
    {
    	System.out.print(p);
    }
    static void p(int p)
    {
    	System.out.print(p);
    }
    static void p(double p)
    {
    	System.out.print(p);
    }
    static void p(long p)
    {
    	System.out.print(p);
    }
    static void p(char p)
    {
    	System.out.print(p);
    }
    static void p(boolean p)
    {
    	System.out.print(p);
    }
    static void pl(String pl)
    {
    	System.out.println(pl);
    }
    static void pl(int pl)
    {
    	System.out.println(pl);
    }
    static void pl(char pl)
    {
    	System.out.println(pl);
    }
    static void pl(double pl)
    {
    	System.out.println(pl);
    }
    static void pl(long pl)
    {
    	System.out.println(pl);
    }
    static void pl(boolean pl)
    {
    	System.out.println(pl);
    }
    static void pl()
    {
    	System.out.println();
    }
    static int[] ari(int n)throws IOException
    {
    	int ar[]=new int[n];
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	for(int x=0;x<n;x++)
    		ar[x]=Integer.parseInt(st.nextToken());
    	return ar;
    }
    static int[][] ari(int n,int m)throws IOException
    {
    	int ar[][]=new int[n][m];
    	for(int x=0;x<n;x++)
    	{
    		st=new StringTokenizer(br.readLine());
    		for(int y=0;y<m;y++)
    			ar[x][y]=Integer.parseInt(st.nextToken());
    	}
    	return ar;
    }
    static long[] arl(int n)throws IOException
    {
    	long ar[]=new long[n];
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	for(int x=0;x<n;x++)
    		ar[x]=Long.parseLong(st.nextToken());
    	return ar;
    }
    static long[][] arl(int n,int m)throws IOException
    {
    	long ar[][]=new long[n][m];
    	for(int x=0;x<n;x++)
    	{
    		st=new StringTokenizer(br.readLine());
    		for(int y=0;y<m;y++)
    			ar[x][y]=Long.parseLong(st.nextToken());
    	}
    	return ar;
    }
    static String[] ars(int n)throws IOException
    {
    	String ar[]=new String[n];
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	for(int x=0;x<n;x++)
    		ar[x]=st.nextToken();
    	return ar;
    }
    static String[][] ars(int n,int m)throws IOException
    {
    	String ar[][]=new String[n][m];
    	for(int x=0;x<n;x++)
    	{
    		st=new StringTokenizer(br.readLine());
    		for(int y=0;y<m;y++)
    			ar[x][y]=st.nextToken();
    	}
    	return ar;
    }
    static double[] ard(int n)throws IOException
    {
    	double ar[]=new double[n];
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	for(int x=0;x<n;x++)
    		ar[x]=Double.parseDouble(st.nextToken());
    	return ar;
    }
    static double[][] ard(int n,int m)throws IOException
    {
    	double ar[][]=new double[n][m];
    	for(int x=0;x<n;x++)
    	{
    		st=new StringTokenizer(br.readLine());
    		for(int y=0;y<m;y++)
    			ar[x][y]=Double.parseDouble(st.nextToken());
    	}
    	return ar;
    }
    static char[] arc(int n)throws IOException
    {
    	char ar[]=new char[n];
    	if(!st.hasMoreTokens())
    		st=new StringTokenizer(br.readLine());
    	for(int x=0;x<n;x++)
    		ar[x]=st.nextToken().charAt(0);
    	return ar;
    }
    static char[][] arc(int n,int m)throws IOException
    {
    	char ar[][]=new char[n][m];
    	for(int x=0;x<n;x++)
    	{
    		st=new StringTokenizer(br.readLine());
    		for(int y=0;y<m;y++)
    			ar[x][y]=st.nextToken().charAt(0);
    	}
    	return ar;
    }
    static void par(int ar[])
    {
    	for(int a:ar)
    		System.out.print(a+" ");
    	System.out.println();
    }
    static void par(int ar[][])
    {
    	for(int a[]:ar)
    	{
    		for(int aa:a)
    			System.out.print(aa+" ");
    		System.out.println();
    	}
    }
    static void par(long ar[])
    {
    	for(long a:ar)
    		System.out.print(a+" ");
    	System.out.println();
    }
    static void par(long ar[][])
    {
    	for(long a[]:ar)
    	{
    		for(long aa:a)
    			System.out.print(aa+" ");
    		System.out.println();
    	}
    }
    static void par(String ar[])
    {
    	for(String a:ar)
    		System.out.print(a+" ");
    	System.out.println();
    }
    static void par(String ar[][])
    {
    	for(String a[]:ar)
    	{
    		for(String aa:a)
    			System.out.print(aa+" ");
    		System.out.println();
    	}
    }
    static void par(double ar[])
    {
    	for(double a:ar)
    		System.out.print(a+" ");
    	System.out.println();
    }
    static void par(double ar[][])
    {
    	for(double a[]:ar)
    	{
    		for(double aa:a)
    			System.out.print(aa+" ");
    		System.out.println();
    	}
    }
    static void par(char ar[])
    {
    	for(char a:ar)
    		System.out.print(a+" ");
    	System.out.println();
    }
    static void par(char ar[][])
    {
    	for(char a[]:ar)
    	{
    		for(char aa:a)
    			System.out.print(aa+" ");
    		System.out.println();
    	}
    }
    static public void main(String[] args)throws Exception{
    	st=new StringTokenizer(br.readLine());
    	long a=l();
    	long b=l();
    	long n = g(a, b); 
        int c=0;
        long s=n;
        
        
        HashSet<Long> l=new HashSet<>(100);
        l.add(1l);
        while (n%2==0) 
        { 
            l.add(2l); 
            n /= 2; 
        }
        for (long i = 3; i <=(long)Math.sqrt(n); i+= 2) 
        {  
            while (n%i == 0) 
            { 
                l.add(i);
                n /= i; 
            } 
        }
        if (n > 2) l.add(n);
        
        
       
        
        
        
        //l.add(s);
    	//pl(l+"");
    	pl(l.size());
    }
    static long g(long a, long b) 
    { 
        if (a == 0l)  
            return b;    
        return g(b%a,a); 
    }
}
class ar implements Comparable<ar>
{
	int ar[]=new int[2];
	ar(int arr[])
	{
		ar[0]=arr[0];
		ar[1]=arr[1];
	}
	public int compareTo(ar o)
	{
		if(this.ar[0]>o.ar[0])return 1;
		if(this.ar[0]<o.ar[0])return -11;
		if(this.ar[1]>o.ar[1])return 1;
		if(this.ar[1]<o.ar[1])return -1;
		return 0;
	}
}

