#include<vector>
#include<iostream>
#include<queue>
#include<functional>
using namespace std;

constexpr int INF = 1e9;

struct edge
{
	int from, to, cost;
	bool operator>(const edge& other)
	const{
		return cost > other.cost;
	}
};

using edges= vector<edge>;
using graph = vector<edges>;

void add_edge(graph& g,int from,int to,int cost)
{
	g[from].push_back(edge{ from,to,cost });
	g[to].push_back(edge{ to,from,cost });
}

edges prim(const graph& g)
{
	const int N = g.size();
	vector<bool>used(N);
	used[0] = true;
	priority_queue<edge, vector<edge>, greater<edge>>que;
	for (auto e:g[0])
	{
		que.push(e);
	}
	edges mst;
	while (!que.empty())
	{
		auto e = que.top();
		que.pop();
		if (used[e.to])
		{
			continue;
		}
		mst.push_back(e);
		used[e.to] = true;
		for (auto new_e:g[e.to])
		{
			que.push(new_e);
		}


	}
	return mst;


	
}
int main()
{
	int V, E;
	cin >> V >> E;
	graph g(V);
	for (int i = 0; i < E; i++)
	{
		int s, t, w;
		cin >> s >> t >> w;
		add_edge(g, s, t, w);
	}
	const auto mst = prim(g);
	int ans = 0;
	for (auto& e : mst)
	{
		ans += e.cost;
	}
	cout << ans << endl;
	return 0;
}
/*
メンバ関数（今回は演算子オーバーロードですが同じ）のこの位置にある const の意味は
この関数の呼び出しによって，オブジェクトの内部状態が変わらないという保証になります
つまり今回であれば，operator > が呼び出されても，edge の中身のデータは変わらないよ，ということを，関数の宣言時点で明言しているということです
エラーが出る理由についてですが
std::sort のデフォルトの挙動では，std::less が使われます
（つまり < ですね
std::less<edge> の呼び出し時の引数の型は (const edge& e1, const edge& e2) であり，どちらも const が付いています
つまり，std::less の中で，e1 にも e2 にも変更を加えることが出来ないということを意味します．
もしこの時に，e1 （あるいは e2）の const でないメンバ関数を呼び出すことが出来てしまうと，オブジェクトの状態が変わる可能性があり，e1 が const であることに反し許容できません．
したがってエラーとなります．
const をつけた場合，その関数を呼び出しても内部状態が変わらないことがコンパイラにわかるので，std::less<edge> のなかで e1 < e2 が呼び出されても，問題ないというわけです
多分さっきのコードは greater だったはずなので，less を greater に置き換えて貰えれば同じです
ひえ～長文になってしまった（C++早口オタクじゃん（ごめん
doi [19:01]
　less の引数のe1あるいはe2がconstであることはbool operator>(const edge& other)で引数がconstであることによってしめされないのでしょうか？
suibaka [19:02]
e1 > e2 としたとき，other は e2 になってこっちはいいのですが，e1 はそうじゃないのでダメですね
例えば operator の定義を edge の外に出して，
bool operator>(const edge& e1, const edge& e2) {
return e1.cost > e2.cost;
}
とやれば問題ないです．
suibaka [19:33]
メンバ関数は，暗黙のうちに（僕らには見えないけど）第一引数にそのオブジェクトのポインタが渡されています
w.f(42) なら f(&w, 42) みたいなイメージです
ここまでは大丈夫ですか
doi [19:34]
大丈夫です
suibaka [19:35]
メンバ関数を const 修飾すると，この this ポインタに const が付きます
すると，this ポインタが指す先のデータの変更がダメという意味になりますね
f() は const 修飾されているので，this->data = arg は this が const なのでできません．
g は問題ないです．this は const じゃないから．
#include <iostream>
using namespace std;
class widget {
public:
widget() : data(0) {}
void f(int arg) const {
data = arg; // error
}
void g(int arg) {
data = arg; // ok
}
private:
int data;
};
// widget::f, g はこういう意味だと思うと良い
// void f(const this ポインタ, int arg) {
//	this->data = arg;
//}
// void g(widget* w, int arg) {
// this->data = arg;
//}
//
// メンバ関数の第1引数には，暗黙のうちに this ポインタが渡されている．
// メンバ関数を const 修飾すると，この this ポインタに const がつく．
// だから，this->data = arg は const 修飾されているとエラーになる．
// g の方は const じゃないので，this->data はOK
int main() {
widget w;
w.f(42); // error
w.g(42); // ok
return 0;
}
*/
