import java.io.*;

public class Main{
	public static void main(String[] args){
		BufferedReader reader=new BufferedReader(new InputStreamReader(System.in));
		try{
			String s,t;
			while((s=reader.readLine()) !=null &&(t=reader.readLine()) !=null){
				double n=Integer.parseInt(s);
				double m=Integer.parseInt(t);
				if(n ==0 &&m==0){
					break;
				}
				double x=n*n;
				double l=Math.sqrt(m*m+n*n/4);
				double y=2*n*l;
				System.out.println(x+y);
			}
		}catch(IOException e){
			System.exit(0);
		}
	}
}