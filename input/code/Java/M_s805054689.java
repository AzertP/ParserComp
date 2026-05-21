import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

class Main
{
    static int mod=1000000007;
    static long dp[][];
    public static void main(String[] args) throws IOException {
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
        int n=Integer.parseInt(br.readLine());
        List<List<Integer>> list=new ArrayList<>();
        dp=new long[n+1][2];
        for(int i=0;i<dp.length;i++)
            for(int j=0;j<dp[0].length;j++)
                dp[i][j]=-1;
        for(int i=0;i<=n;i++)
        {
            list.add(new ArrayList<>());
        }
        for(int l=0;l<n-1;l++)
        {
            String str[]=br.readLine().split(" ");
            int a=Integer.parseInt(str[0]);
            int b=Integer.parseInt(str[1]);
            list.get(b).add(a);
            list.get(a).add(b);
        }
        System.out.println(calculate(list,1,0,-1));

    }

    private static long calculate(List<List<Integer>> list, int index, int black,int parent) {
        long res=1;
        long ans=0;
        if(dp[index][black]!=-1)
            return dp[index][black]%mod;
        for(int child:list.get(index))
        {
            if(child!=parent)
                res=((res%mod)*(calculate(list,child,0,index)%mod))%mod;
        }
        ans=((ans%mod)+(res%mod))%mod;
        if(black==0)
        {
            res=1;
            for(int child:list.get(index))
            {
                if(child!=parent)
                    res=((res%mod)*(calculate(list,child,1,index)%mod))%mod;
            }
            ans=((ans%mod)+(res%mod))%mod;

        }
        return dp[index][black]=ans%mod;
    }
}