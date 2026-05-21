import java.util.*;
public class Main {
	public static void main(String args[]){
    	Scanner s = new Scanner(System.in);
    	int N = s.nextInt();
      	int arr[] = new int[N];
      	for(int i=0;i<N;i++)
          	arr[i] = s.nextInt();
      	int ans = frog(arr,N);
      	System.out.println(ans);
    }
  	public static int frog(int[] arr, int N){
    	if(N<=1)
          return 0;
      	int dp[] = new int[N];
      	dp[0] = 0;
      	dp[1] = Math.abs(arr[1]-arr[0]);
      	for(int i=2;i<N;i++){
        	int ans1 = Math.abs(arr[i]-arr[i-1])+dp[i-1];
          	int ans2 = Math.abs(arr[i]-arr[i-2])+dp[i-2];
          	dp[i] = Math.min(ans1,ans2);
        }
      	return dp[N-1];
    }
}