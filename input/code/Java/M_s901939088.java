import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Scanner insert = new Scanner(System.in);
		int n = -1;
		int m = -1;
		while(true) {
			String line = insert.nextLine();
			String array[] = line.split(" ");
			n = Integer.parseInt(array[0]);
			m = Integer.parseInt(array[1]);
			if((n==0) && (m == 0)) {
				break;
			}
			ArrayList<Integer> sums = new ArrayList<Integer>();
			int[] input = new int[n];
			line = insert.nextLine();
			String array2[] = line.split(" ");
			for(int i =0; i < n; i++) {
				input[i] = (Integer.parseInt(array2[i]));
			}
			for(int i = 0; i < (n - 1); i ++) {
				for(int j = (i + 1); j < n; j++) {
					sums.add(input[i] + input[j]);
				}
			}
			Collections.sort(sums,Comparator.reverseOrder());
			for(int i = 0; i < sums.size(); i++) {

				if(sums.get(i) <= m) {
					result.add(sums.get(i));
					break;
				}
				if(i == (sums.size() - 1)) {
					result.add(-1);
				}
			}

		}
		for(int i = 0;i < (result.size());i++) {
			if(result.get(i) == -1) {
				System.out.println("NONE");
			}
			else {
				System.out.println(result.get(i));
			}
		}
	}
}

