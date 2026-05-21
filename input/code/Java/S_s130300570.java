import java.util.Scanner;

public class Main {
	
	public static void main(String[] args) throws Exception {
	    Scanner sc = new Scanner(System.in);   
	    
	    int A = sc.nextInt();
	    int B = sc.nextInt();
	    int C = sc.nextInt();
	    int D = sc.nextInt();
	    
	    if (A+B == C+D) {
	    	System.out.println("Balanced");
	    } else if (A+B > C+D) {
	    	System.out.println("Left");
	    } else {
	    	System.out.println("Right");
	    }
	}    
}	