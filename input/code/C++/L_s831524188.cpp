#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <stdlib.h>
#include <string>
#include <vector>

#define INF 1000000000
#define MOD 1000000007
#define ll long long
#define rep(i,a,b) for(int i = (a); i < (b); ++i)
#define bitget(a,b) (((a) >> (b)) & 1)
#define vint vector<int>
#define ALL(x) (x).begin(),(x).end()
#define C(x) cout << #x << " : " << x << endl
#define scanf scanf_s

using int32 = int_fast32_t;
using int64 = int_fast64_t;
using uint32 = uint_fast32_t;
using uint64 = uint_fast64_t;

using namespace std;

template<typename Monoid>
class SegTree {
public:
	Monoid* tree;
	size_t size;
	int32 i;
	//?????????????????????
	SegTree(size_t length) {
		--length;
		size = length & 0xffff0000 ? length & 0xffff0000 : length;
		size = size & 0xff00ff00 ? size & 0xff00ff00 : size;
		size = size & 0xf0f0f0f0 ? size & 0xf0f0f0f0 : size;
		size = size & 0xcccccccc ? size & 0xcccccccc : size;
		size = size & 0xaaaaaaaa ? size & 0xaaaaaaaa : size;
		size = length ? size << 1 : 1;
		++length;
		tree = (Monoid *)calloc(size << 1, sizeof(Monoid));
	}
	//?????????
	void set() {
		size <<= 1;
		for (i = 0;i < size;++i) tree[i].set();
		size >>= 1;
	}
	//tree[index]???data?????´??°
	void update(size_t index, Monoid &data) {
		index += size;
		tree[index] = data;
		while (index >>= 1) tree[index] = tree[index << 1] + tree[index * 2 + 1];
	}
	//[begin,end)??????????????????
	Monoid range(size_t begin, size_t end) {
		Monoid retL, retR;
		for (begin += size, end += size;begin < end;begin >>= 1, end >>= 1) {
			if (begin & 1) retL = retL + tree[begin++];
			if (end & 1)  retR = tree[--end] + retR;
		}
		return retL + retR;
	}
};

struct MIN {
	int32 e;
	MIN() { set(); }
	MIN(int32 x) { e = x; }
	void set() { e = 2147483647; }
	void update(MIN b) { e = b.e; }
	MIN operator+(MIN &other) {
		return e < other.e ? MIN(e) : MIN(other.e);
	};
};

int main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);
	//*
	int32 n, q;
	cin >> n >> q;
	int32 c, x,y1;
	MIN y2;
	SegTree<MIN> t(n);
	t.set();
	rep(i, 0, q) {
		cin >> c;
		if (c) {
			cin >> x >> y1;
			printf("%d\n", t.range(x, y1 + 1).e);
		}
		else {
			cin >> x >> y2.e;
			t.update(x, y2);
		}
	}
	//*/
	return 0;
}