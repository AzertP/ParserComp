import java.util.*;

class Main{
    public static void main(String[] args){
	Scanner s = new Scanner(System.in);
	int n, a;
	int[][] cards;
	cards = new int[4][14];
	boolean[][] flag;
	flag = new boolean[4][14];
	char c;
	
	for(int i = 0 ; i < 4 ; i++){
	    for(int j = 1 ; j < 14 ; j++){
		cards[i][j] = j;
		flag[i][j] = false;
	    }
	}
	
	n = s.nextInt();
	for(int i = 0 ; i < n ; i++){
	    c = s.next().charAt(0);
	    a = s.nextInt();
	    if(c == 'S') flag[0][a] = true;
	    if(c == 'H') flag[1][a] = true;
	    if(c == 'C') flag[2][a] = true;
	    if(c == 'D') flag[3][a] = true;
	}
	
	for(int i = 0 ; i < 4 ; i++){
	    for(int j = 1 ; j < 14 ; j++){
		if(flag[i][j] == false){
		    if(i == 0) System.out.printf("S %d\n", j);
		    if(i == 1) System.out.printf("H %d\n", j);
		    if(i == 2) System.out.printf("C %d\n", j);
		    if(i == 3) System.out.printf("D %d\n", j);
		}
	    }
	}
    }
}