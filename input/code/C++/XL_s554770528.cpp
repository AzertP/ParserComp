#include <bits/stdc++.h>
using namespace std;
// Macros for easier access
#define ll long long
#define umap unordered_map
#define uset unordered_set
#define test ll cases; cin>>cases; for(ll testCase = 1; testCase <= cases; testCase++)   //test cases
#define rep(i, begin, end, upd)  for(ll i = begin; ((begin < end && upd > 0 && i < end) || (begin > end && upd < 0 && i >= end)); i+=upd)
#define fill(name, val) memset(name, val, sizeof(name));
#define mop(a, op, b)	(a%mod op b%mod)%mod
#define error(x) fixed<<setprecision(x) //cout<<error(5)<<someDouble    -> example - 3.14159
#define vll vector<ll>
#define vvll vector<vll>
#define pll pair<ll, ll>
#define get(array) for(ll index = 0; index < sizeof(array)/sizeof(array[0]); index++) cin>>array[index];
#define boost ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)
#define debug(x) cerr << #x << " : " << (x) << endl
// Constants
#define MX 100001
#define mod 1000000007LL
#define inf 1000000000000000000LL
ll gcd(ll a, ll b){
	while(b){
		a %= b;
		swap(a, b);
	}
	return a;
}
ll power(ll x, ll n){
	x %= mod;	ll res = 1;
	while(n){
		if(n&1)	res = mop(res, *, x);
		x = mop(x, *, x);
		n >>= 1;
	}
	return res;
}
ll fermatInv(ll x){
    if(gcd(x, mod) != 1)  return -1;
    return power(x, mod-2);
}
vll primeSieve(ll n = MX){
	vll primes;
    bool nums[n];   fill(nums, true);
    primes.push_back(2);
    rep(i, 4, n, 2)   nums[i] = false;
    rep(i, 3, n, 2){
        if(nums[i]){
            primes.push_back(i);
            rep(j, i*i, n, i) nums[j] = false;
        }
    }
	return primes;
}
vll factorize(ll n){
    vll res;
    ll rootn = sqrt(n);
    rep(i, 1, rootn+1, 1){
        if(n % i == 0){
            res.push_back(i);
            if(n/i != i)    res.push_back(n/i);
        }
    }
    return res;
}
vvll idMatrix(ll n){
	vvll id;
	rep(i, 0, n, 1){
		id.push_back({});
		rep(j, 0, n, 1)
			id[i].push_back(i==j);
	}
	return id;
}
vvll matrixMult(vvll mx1, vvll mx2){
	ll r = mx1.size(), c1 = mx2.size(), c = mx2[0].size();
	vvll result(r);
	rep(i, 0, r, 1)
		rep(j, 0, c, 1){
			ll sum = 0;
			rep(k, 0, c1, 1)
				sum = mop(sum, +, mop(mx1[i][k], *, mx2[k][j]));
			result[i].push_back(sum);
		}
	return result;
}
vvll matrixExpo(vvll matrix, ll n){
	if(n == 0)	return idMatrix(matrix.size());
	vvll sa = matrixExpo(matrix, n/2);
	sa = matrixMult(sa, sa);
	if(n & 1)	return matrixMult(sa, matrix);
	else		return sa;
}
ll nCr(ll n, ll r){
	if(r == 0)	return 1;
	ll f[n+1];	f[0] = 1;
	for(ll i = 1; i <= n; i++)	f[i] = mop(f[i-1], *, i);
	return mop(f[n], *, mop(fermatInv(f[r]), *, fermatInv(f[n-r])));
}
ll nCrDP(ll n, ll r, ll p){
	r = min(r, n - r);
	ll C[r+1];	fill(C, 0);
	C[0] = 1;
	for(ll i = 1; i <= n; i++)
		for(ll j = min(i, r); j > 0; j--)
			C[j] = (C[j] + C[j-1]) % p;
	return C[r];
}
ll smallpnCr(ll n, ll r, ll p){
	if(r == 0)	return 1;
	ll ni = n % p, ri = r % p;
	return (smallpnCr(n/p, r/p, p) * nCrDP(ni, ri, p)) % p;
}
ll linearRec(vll f, vll c, ll n){
	ll order = f.size();
	vvll T = idMatrix(order);
	for(ll i = 0; i < order-1; i++)
		for(ll j = 0; j < order; j++)
			T[i][j] = T[i+1][j];
	for(ll i = 0; i < order; i++)	T[order-1][i] = c[i];
	T = matrixExpo(T, n - order);
	vvll f0;	for(auto x : f)	f0.push_back({x});
	T = matrixMult(T, f0);
	return T[order-1][0];
}
class TreeNode{
    public:
    ll val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(ll v){
        val = v;
        left = right = NULL;
    }
};
class STree{
	public:
	ll n;
	ll* tree;
	ll queryBase;
	ll segOp(ll left, ll right);
	STree(ll arrLen, ll* arr, ll qb){	//build the tree
		n = arrLen;
		queryBase = qb;
		tree = new ll[2*n];
		rep(i, 0, n, 1)	tree[n+i] = arr[i];
		rep(i, n-1, 0, -1)	tree[i] = segOp(tree[i<<1], tree[i<<1|1]);
	}
	void update(ll index, ll value){	//point update
		for(tree[index += n] = value; index > 1; index >>= 1)	tree[index>>1] = segOp(tree[index], tree[index^1]);
	}
	ll query(ll left, ll right){	// [left, right)
		ll lres = queryBase, rres = queryBase;
		for(left += n, right += n; left < right; left >>= 1, right >>= 1){
			if(left&1) lres = segOp(lres, tree[left++]);
			if(right&1) rres = segOp(rres, tree[--right]);
		}
		return segOp(lres, rres);
	}
};
class Graph{
    ll N;
    bool weighted;
	bool directed;
    umap<ll, umap<ll, ll>> g; //(u, v, w) Assume all graphs are weighted. Unweighted => every edge weight = 1
	public:
	class Edge{
	public:
		ll u, v, w;
		Edge(ll U, ll V, ll W){
			tie(u, v, w) = tie(U, V, W);
		}
		bool operator<(Edge const& other){
			return w < other.w;
		}
	};
    Graph(ll n, bool w = false, bool d = false){
        N = n;
        weighted = w;
		directed = d;
    }
    auto addEdge(ll u, ll v, ll w = 1){
        g[u][v] = w;
		g[v];
		if(!directed)	g[v][u] = w;
    }
    auto printEdges(){
        for(auto u : g){
            cout<<u.first<<"-> ";
            for(auto v : u.second){
                cout<<v.first;
                if(weighted) cout<<"("<<v.second<<")";
                cout<<' ';
            }
            cout<<endl;
        }
    }
	auto dijkstra(ll src){
		umap<ll, ll> dist;
		set<pll> s;
		for(auto i : g)	dist[i.first] = inf;
		dist[src] = 0;
		s.insert({0, src});
		while(s.size()){
			pll temp = *(s.begin());
			s.erase(s.begin());
			ll u = temp.second;
			auto du = &dist[u];
			for(auto i : g[u]){
				ll v, w;
				tie(v, w) = i;
				auto dv = &dist[v];
				if(*dv > *du + w){
					if(*dv != inf)	s.erase(s.find({*dv, v}));
					*dv = *du + w;
					s.insert({*dv, v});
				}
			}
		}
		return dist;
	}
	auto flowar(){	// Floyd Warshall
		umap<ll, umap<ll, ll>> dist;
		vll nodes;
		for(auto x : g)	nodes.push_back(x.first);
		for(auto x : nodes)
			for(auto y : nodes)
				if(x == y)	dist[x][y] = 0;
				else if(g[x].find(y) == g[x].end())	dist[x][y] = inf;
				else	dist[x][y] = g[x][y];
		for(auto k : nodes)
			for(auto i : nodes)
				for(auto j : nodes)
					dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
		return dist;
	}
	auto MST(){	//Kruskal's
		if(directed){
			cerr<<"Attempted Kruskal MST on directed graph!!!"<<endl;
			exit(0);
		}
		umap<ll, ll> parent;
		umap<ll, ll> rank;
		auto make_set = [&parent, &rank](ll v){
			parent[v] = v;
			rank[v] = 0;
		};
		function<ll(ll)> find_set;
		find_set = [&parent, &find_set](ll v) -> ll{
			if(v == parent[v])	return v;
			return parent[v] = find_set(parent[v]);
		};
		auto union_set = [&parent, &rank, &find_set](ll a, ll b){
			a = find_set(a);
			b = find_set(b);
			if(a == b)	return;
			if(rank[a] < rank[b])	swap(a, b);
			parent[b] = a;
			rank[a] += (rank[a] == rank[b]);
		};
		vector<Edge> edges;
		vector<Edge> mst;
		for(auto x : g){
			make_set(x.first);
			for(auto y : x.second){
				if(y.first < x.first){
					Edge e = Edge(x.first, y.first, y.second);
					edges.push_back(e);
				}
			}
		}
		sort(edges.begin(), edges.end());
		for(auto e : edges){
			if(find_set(e.u) != find_set(e.v)){
				mst.push_back(e);
				union_set(e.u, e.v);
			}
		}
		return mst;
	}
	auto dfs(ll curr, vll& res, uset<ll>& vis){
		if(vis.find(curr) != vis.end())	return;
		vis.insert(curr);
		for(auto x : g[curr])
			dfs(x.first, res, vis);
		res.push_back(curr);
	}
	auto toposort(){
		if(!directed){
			cerr<<"Toposort on bidirectional graph!!!";
			exit(0);
		}
		vll top;	uset<ll> vis;
		for(auto x : g)
			if(vis.find(x.first) == vis.end())
				dfs(x.first, top, vis);
		reverse(top.begin(), top.end());
		return top;
	}
};

/*  =======TL;DR=======
    Author : zenolus
    TreeNode (binary tree):
		TreeNode(v)   => create new node with val v
    Graph (Adjacency List representation):
		Graph(nodes, weighted = false, directed = false)  => create a graph with n nodes, specify if weighted
		addEdge(u, v, weight = 1)   => add an edge in graph, specify weight if necessary
		printEdges() => display the adjacency list
		dijkstra(src)	=> single src all dest shortest distance. returns a map <node, dist> Ot(ElogV) Os(V)	use AUTO
		flowar()	=> All pair shortest distance. returns 2D map <u, <v, d>>	Ot(V^3) Os(V^2)	use AUTO
		MST()	=> Returns the edge list containing N - 1 edges forming the Kruskal's MST	Ot(ElogE) Os(V+E)	use AUTO
		toposort()	=> returns dfs based topological sorted vector for DAGs		Ot(V+E) Os(V)	use AUTO
    STree (Segment Tree):
		STree(arrayLength, array, queryBase)	=> create the segment tree
		update(index, value)	=> point update
		query(left, right)	=> query segment operation on [l, r)
	Algorithms :
		gcd(a, b)   => returns gcd of a and b
		power(x, n) => return x raised to power n in O(log n) time, if required, specify modulo
		fermatInv(x)  => returns modular multiplicative inverse of x with mod
		primeSieve(n = MX)   => returns a vector of all the primes till n
		factorize(n)    => returns a vector of factors of n
		modMult(a, b)	=> returns (a*b)%mod
		idMatrix(matrix)	=> returns an identity matrix of dimensions similar to passed matrix
		matrixMult(A, B)	=> returns (A.B)%mod
		matrixExpo(A, n)	=> returns A power n % mod
		nCr(n, r)  => returns nCr value % mod using Fermat's theorem. O(n + log p)
		nCrDP(n, r, p)	=> returns nCr % p for small n, r, p	Ot(n*r), Os(n)
		smallnCr(n, r, p)	=> returns nCr % p for small p using Lucas' theorem.	Ot(p^2 + log n base p), Os(p)
		linearRec(f, c, n)	=> returns nth term of linear recurrence. F(n) = c1F(n-1) + c2F(n-2) + ... + ckF(n-k)
							Ex: linearRec({1, 2, 3}, {3, 4, 5}, 8)	=> starting terms 1, 2, 3. c1 = 5, c2 = 5, c3 = 3. k = 3
*/

ll STree::segOp(ll left, ll right){
	return left + right;	//Change per requirement
}

ll n;
ll A[MX], B[MX], C[MX];
ll dp[MX][4];
ll solve(ll i, ll pc = 0){
	if(i == n)	return 0;
	auto d = &dp[i][pc];
	if(*d != -1)	return *d;
	ll ans = 0;
	for(ll c = 1; c <= 3; c++){
		if(pc == c)	continue;
		ll val = c == 1 ? A[i] : c == 2 ? B[i] : c == 3 ? C[i] : 0;
		ans = max(ans, val + solve(i+1, c));
	}
	return *d = ans;
}
int main(){
	boost;
	cin>>n;
	for(ll i = 0; i < n; i++)	cin>>A[i]>>B[i]>>C[i];
	fill(dp, -1);
	cout<<solve(0);
}