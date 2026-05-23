using namespace std;

typedef pair<int,int> P;

int n;
int d[20][20];
int dp[1<<15][20];
bool use[20];
vector<string> s;

int rec(int st, int prv){
  if(st==(1<<n)-1)return 0;
  if(prv>=0 && dp[st][prv]>=0)return dp[st][prv];

  int res = 10000;
   
  for(int i=0;i<n;i++){
    if( (st>>i) & 1 )continue;
    int cost = s[i].size();
    if(prv>=0)cost -= d[prv][i];
    res = min(res,rec(st|(1<<i),i)+cost);
  }
  return dp[st][prv] = res;
}

int main(){
  while(cin >> n,n){
    s.resize(n);
    for(int i=0;i<n;i++){
      cin >> s[i];
      use[i] = false;
    }

    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++){
	if(i==j)continue;
	if(s[i].size() > s[j].size())continue;
	int l = s[i].size();
	for(int k=0;k+l<=s[j].size();k++){
	  if(s[i] == s[j].substr(k,l)){
	    use[i] = true;
	    break;
	  }
	}
      }
    }

    int p = 0;
    for(int i=0;i<n;i++){
      if(use[i])s.erase(s.begin()+p);
      else p++;
    }

    n = s.size();

    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++){
	if(i==j)continue;
	d[i][j] = 0;
	int l = min(s[i].size(),s[j].size()) - 1;
	for(int k=l;k>=1;k--){
	  if(s[i].substr(s[i].size()-k,k) == s[j].substr(0,k)){
	    d[i][j] = k;
	    break;
	  }
	}
      }
    }

    for(int i=0;i<(1<<n);i++){
      for(int j=0;j<n;j++)dp[i][j] = -1;
    }
    
    cout << rec(0,-1) << endl;
  }
}
