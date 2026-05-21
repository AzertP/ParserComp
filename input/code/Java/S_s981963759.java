import java.util.Scanner;

class Main{

	public static void main(String args[]){
		Scanner sc = new Scanner(System.in);
		long N = sc.nextLong();
		long K = sc.nextLong();

		long ans = 0;

		if(N>K){
			long x = N / K;
			long ans1 = N - (K * x);
			long ans2 = ((x + 1) * K) - N;
			ans = (ans1 > ans2)? ans2 : ans1;
		}else if(N<K){
			long ans1 = K - N;
			long ans2 = N;
			ans = (ans1 > ans2)? ans2 : ans1;
		}else{
			ans = 0;
		}

		System.out.println(ans);
		// System.out.println(S.toString());

	}
}