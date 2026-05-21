import java.util.*;


public class Main {
	public static void main(String args[]){
		Scanner in = new Scanner(System.in);
		while(in.hasNext()){
			try{
				double a = in.nextInt();
				double b = in.nextInt();	
				double S = (a * b)/3.305785; 
				System.out.println(S);
			}catch(Exception e){
				System.exit(0);
			}
		}
	}
}