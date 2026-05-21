import java.util.*;
import java.io.*;
public class Main
{
    static BufferedWriter bw;
    static BufferedReader br;
    static void solve()throws Exception
    {
        int r = Integer.parseInt(br.readLine());
        double result = 2 * 22/7.0 * r;
        bw.write(Double.toString(result));
        bw.newLine();
    }
    public static void main(String []args)throws Exception
    {
        br=new BufferedReader(new InputStreamReader(System.in));
        bw=new BufferedWriter(new OutputStreamWriter(System.out));
        // int t = Integer.parseInt(br.readLine());
        // for(int i=0;i<t;i++)
        // {
        //     bw.write("Case #"+(i+1)+": ");
            solve();
        // }
        br.close();
        bw.close();
    }
}
