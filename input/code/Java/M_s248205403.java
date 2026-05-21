import java.util.ArrayList;
import java.util.Scanner;

public class Main {
	Scanner sc;
	public void printH(int[] x){
		for(int i = 0; i < x.length - 1; i++){
			System.out.print(x[i] + " ");
		}
		System.out.println(x[x.length-1]);
	}
	public void run(){
		sc = new Scanner(System.in);
		ArrayList<Integer> n = new ArrayList<Integer>();

		while(sc.hasNextInt()){
			int value = sc.nextInt();
			if(value != 0){
				n.add(value);
			}
			else{
				int value2 = sc.nextInt();
				if(value2 != 0){
					n.add(0);
					n.add(value2);
				}
				else{
					break;
				}
			}
		}
		for(int i = 0; i < n.size(); i = i + 2){
			if(n.get(i) > n.get(i+1)){
				int k = n.get(i);
				n.set(i, n.get(i+1));
				n.set(i+1, k);
			}
		}
		for(int i = 0; i < n.size(); i = i + 2){
			System.out.println(n.get(i) + " " + n.get(i+1));
		}
	}
	public static void main(String[] args){
		new Main().run();
	}
}