import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

//Reindeer with no sense of direction
public class Main{

	int w, h, N, GS, R = 8;
	int[][] map;
	int[][] d = {{-1,0},{0,1},{1,0},{0,-1}};
	List<Integer>[][] adj;
	Map<Integer, Integer>[] dp;
	
	int get(int v, int S, int rest){
		if(rest==1&&S==(1<<N))return 1;
		if(R<=rest&&dp[v].containsKey(S))return dp[v].get(S);
		int res = 0;
		for(int k=0;k<4;k++)for(int nv:adj[v][k]){
			if(((S>>nv)&1)==0)continue;
			if(nv==N&&3<=rest)continue;
			res+=get(nv, S-(1<<v), rest-1);
			break;
		}
		if(R<=rest)dp[v].put(S, res);
		return res;
	}
	
	@SuppressWarnings("unchecked")
	void run(){
		Scanner sc = new Scanner(System.in);
		for(;;){
			w = sc.nextInt(); h = sc.nextInt();
			if((w|h)==0)break;
			N = 0;
			map = new int[h][w];
			for(int i=0;i<h;i++)for(int j=0;j<w;j++){
				map[i][j] = sc.nextInt();
				if(map[i][j]==1)N++;
				else if(map[i][j]==0)map[i][j]=-1;
			}
			int id = 0;
			for(int i=0;i<h;i++)for(int j=0;j<w;j++){
				if(map[i][j]==-1)continue;
				if(map[i][j]==1){
					map[i][j]=id++;
				}
				else {
					map[i][j] = N;
				}
			}
			adj = new List[N+1][4];
			for(int i=0;i<=N;i++)for(int j=0;j<4;j++)adj[i][j]=new ArrayList<Integer>();
			for(int i=0;i<h;i++)for(int j=0;j<w;j++)if(map[i][j]!=-1){
				for(int k=0;k<4;k++){
					int pi = i+d[k][0], pj = j+d[k][1];
					while(0<=pi&&pi<h&&0<=pj&&pj<w){
						if(map[pi][pj]!=-1)adj[map[i][j]][k].add(map[pi][pj]);
						pi+=d[k][0]; pj+=d[k][1];
					}
				}
			}
			GS = (1<<(N+1))-1;
			dp = new Map[N+1];
			for(int i=0;i<=N;i++)dp[i]=new HashMap<Integer, Integer>();
			dp[N].put(1<<N, 1);
			int res = 0;
			for(int k=0;k<4;k++)if(!adj[N][k].isEmpty())res+=get(adj[N][k].get(0), GS, N+1);
			System.out.println(res);
		}
	}
	
	public static void main(String[] args) {
		new Main().run();
	}
}