#include <queue>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int maxn = 1100;
const int dx[] = {1, 0, -1 ,0};
const int dy[] = {0, 1, 0, -1};

struct point{
	int x,y;
};

int d[maxn][maxn];
char mp[maxn][maxn];
queue<point> q;

inline point make_point(int x,int y){
	point ret;
	ret.x = x;
	ret.y = y;
	return ret;
}

int main()
{
	int n, m;
	scanf("%d%d",&n,&m);
	for (int i = 1;i <= n;i++){
		scanf("%s",mp[i] + 1);
	}
	for (int i = 1;i <= n;i++){
		for (int j = 1;j <= m;j++){
			if (mp[i][j] == '#'){
				d[i][j] = 1;
				q.push(make_point(i, j));
			}
		}
	}
	while (!q.empty()){
		point u = q.front();
		q.pop();
		for (int k = 0;k < 4;k++){
			int nx = u.x + dx[k];
			int ny = u.y + dy[k];
			if (nx <= 0 || ny <= 0 || nx > n || ny > m) continue;
			if (d[nx][ny] == 0){
				d[nx][ny] = d[u.x][u.y] + 1;
				q.push(make_point(nx, ny));
			}
		}
	}

	int ans = 0;
	for (int i = 1;i <= n;i++){
		for (int j = 1;j <= m;j++){
			ans = max(ans, d[i][j]);
		}
	}
	printf("%d\n", ans - 1);
    return 0;
}

