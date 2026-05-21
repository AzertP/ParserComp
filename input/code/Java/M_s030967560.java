import java.util.*;

public class Main{
	static HashMap<Integer, Numnum> map;
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(System.in);
		map = new HashMap<Integer, Numnum>();
		boolean[] swm = new boolean[300001];
		boolean[] swm2 = new boolean[300001];
		for(int i=1;i<=300000;i++){
			if(i%7==1||i%7==6){
				swm[i]=true;
				swm2[i]=true;
			}
		}
		for (int i = 7; i <= 299999; i += 7) {
			Numnum num1 = new Numnum(i - 1);
			Numnum num2 = new Numnum(i + 1);

			for (int j = 2; num1.mynum * j <= 300000; j++) {
				swm[num1.mynum * j] = swm[num1.mynum * j] ? false :swm[num1.mynum*j];
			}
			for (int j = 2; num2.mynum * j <= 300000; j++) {
				swm[num2.mynum * j] = swm[num2.mynum * j] ? false :swm[num2.mynum*j];
			}

			map.put(num1.mynum, num1);
			map.put(num2.mynum, num2);
		}
		
		for (int i = 6; i <= 300000; i ++) {
			if(swm[i])map.get(i).set.add(i);
			if(!swm[i])continue;
			for (int j = 2; i * j <= 300000; j++) {
				if(swm2[i*j]){
					map.get(i*j).set.add(i);
					if(swm[j])map.get(i*j).set.add(j);
				}
			}
		}

		while (true){
			int n = sc.nextInt();
			if (n == 1)
				break;
			System.out.print(n + ":");
			for (int result : map.get(n).set) {
				if (swm[result])System.out.print(" " + result);
			}
			System.out.println();
		}

	}
}

class Numnum {
	int mynum;
	TreeSet<Integer> set = new TreeSet<Integer>();
	public Numnum(int num) {
		mynum = num;
	}
	
}