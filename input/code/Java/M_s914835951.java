import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Scanner;


public class Main {

	
	// a b c    0 1 2
	// d e f => 3 4 5
	// g h i    6 7 8
	public static final int[][] adj = new int[][]{
		{1, 3}, //a 
		{0, 2, 4}, //b
		{1, 5}, //c
		{0, 4, 6}, //d
		{1, 3, 5, 7}, //e
		{2, 4, 8}, //f
		{3, 7}, //g
		{4, 6, 8}, //h
		{5, 7}, //i
	};
	
	public static int trans(char a){
		return a - 'A';
	}
	
	public static void main(String[] args) throws IOException {
		Scanner sc = new Scanner(System.in);
		
		for(int i = 0; i < 1000; i++){
			String input = sc.nextLine();
			
			char prev = 0;
			boolean flag = true;
			for(char in_ch : input.toCharArray()){
				if(prev != 0){
					final int prev_int = trans(prev);
					final int next_int = trans(in_ch);
				
					final int pos = Arrays.binarySearch(adj[prev_int], next_int);
					
					if(pos < 0){
						flag = false;
						break;
					}
				}
				
				prev = in_ch;
			}
			
			if(flag){
				System.out.println(input);
			}
			
		}
		
		
	}

}