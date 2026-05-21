import java.util.*;
import java.io.*;

public class Main {
    public static void main(String[] args) throws IOException{
        BufferedReader f = new BufferedReader(new InputStreamReader(System.in));

        StringTokenizer st=new StringTokenizer(f.readLine());
        int n=Integer.parseInt(st.nextToken());
        int m=Integer.parseInt(st.nextToken());
        int[]arr=new int[n];
        int[]arr2=new int[m];
        st=new StringTokenizer(f.readLine());
        for(int i=0;i<n;i++){
            arr[i]=Integer.parseInt(st.nextToken());
        }
        st=new StringTokenizer(f.readLine());
        for (int j=0;j<m;j++){
            arr2[j]=Integer.parseInt(st.nextToken());
        }
        long MOD=1000000007;
        long[][]dp=new long[n+1][m+1];
//        prefix[0][0]=1;
        dp[0][0]=1;
        for(int i=0;i<=n;i++){
            prefix[i][0]=1;
        }
        for(int i=0;i<=m;i++)prefix[0][i]=1;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
//                if(i==0 && j==0)continue;
                if(i!=0 && j!=0){
                    prefix[i][j]=+prefix[i-1][j]+prefix[i][j-1]-prefix[i-1][j-1]+prefix[i][j];
                }
                else if(i!=0){
                    prefix[i][j]=prefix[i][j]+prefix[i-1][j];
                }
                else if(j!=0){
                    prefix[i][j]=prefix[i][j]+prefix[i][j-1];
                }
                if(arr[i-1]==arr2[j-1]){
                    dp[i][j]++;
                    dp[i][j]+=query(1,1,i-1,j-1);;

                }
                prefix[i][j]+=dp[i][j];
                prefix[i][j]%=MOD;
                dp[i][j]%=MOD;
                if(prefix[i][j]<0)prefix[i][j]+=MOD;
                if(dp[i][j]<0)dp[i][j]+=MOD;
                prefix[i][j]%=MOD;
                dp[i][j]%=MOD;
            }
        }
        System.out.println(prefix[n][m]);



        f.close();
    }

    static long[][]prefix=new long[2200][2200];
    static long query(int x1,int y1, int x2,int y2){
        if(x2<0){
            return 0;
        }
        if(y2<0){
            return 0;
        }
        long ret=prefix[x2][y2];
        if(x1!=0 && y1!=0){
            ret-=prefix[x1-1][y2];
            ret-=prefix[x2][y1-1];
            ret+=prefix[x1-1][y1-1];
        }
        else if(y1!=0){
            ret-=prefix[x2][y1-1];
        }
        else if (x1!=0){
            ret-=prefix[x1-1][y2];
        }
        return ret;

    }
}
