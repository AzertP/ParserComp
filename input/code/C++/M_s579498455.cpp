using namespace std;

int bin_s(vector<int> &u, int key){
    int ok=-1, ng=u.size();
    while(ok+1<ng){
        int mid=(ok+ng)/2;
        if(u[mid]<key) ok=mid;
        else ng=mid;
    }
    return ng;
}


int main() {
	int N, Q; cin >> N >> Q;
	vector<pair<int, pair<int, int>>> v(N);
	for(int i=0; i<N; ++i){
	    int s, t, x;
	    cin >> s >> t >> x;
	    v[i]=make_pair(x, make_pair(s-x, t-x));
	}
	sort(v.begin(), v.end());
	vector<int> D(Q), next_num(Q), ans(Q, 1+1e9);
	for(int i=0; i<Q; ++i){
	    cin >> D[i];
	    next_num[i]=i+1;
	}
	for(int i=0; i<N; ++i){
	    int left=bin_s(D, v[i].second.first), right=bin_s(D, v[i].second.second);
	    while(left<right){
	        ans[left]=min(v[i].first, ans[left]);
	        int next=next_num[left];
	        next_num[left]=max(right, next_num[left]);
	        left=next;
	    }
	}
	for(int i=0; i<Q; ++i) cout << (ans[i]>1e9 ? -1 : ans[i]) << endl;
	return 0;
}
