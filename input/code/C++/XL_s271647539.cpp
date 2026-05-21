#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include "bits/stdc++.h"
#define rep(i, n) for (int i = 0; i < (n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (n); ++i)
#define rrep(i, n) for (int i = (n) - 1; i >= 0; --i)
#define rfor(i, m, n) for (int i = (m); i >= (n); --i)
#define sz(x) ((int)(x).size())
#define all(x) (x).begin(),(x).end()
#define rall(x) (x).rbegin(),(x).rend()
#define range_it(a, l, r) (a).begin() + (l), (a).begin() + (r)

using namespace std;
using ll = long long;		using ld = long double;
using VB = vector<bool>;	using VVB = vector<VB>;
using VI = vector<int>;		using VVI = vector<VI>;
using VL = vector<ll>;		using VVL = vector<VL>;
using VS = vector<string>;	using VD = vector<ld>;
using PII = pair<int, int>;	using VP = vector<PII>;
using PLL = pair<ll, ll>;	using VPL = vector<PLL>;
template<class T>using PQ = priority_queue<T>;
template<class T>using PQS = priority_queue<T, vector<T>, greater<T>>;
constexpr int inf = (int)1e9;
constexpr ll inf_ll = (ll)1e18, MOD = 1000000007;
constexpr ld PI = M_PI, EPS = 1e-12;

// --- input --- //
#ifdef _WIN32
#define getchar_unlocked _getchar_nolock
#define putchar_unlocked _putchar_nolock
#define fwrite_unlocked fwrite
#define fflush_unlocked _fflush_nolock
#endif
inline int gc() { return getchar_unlocked(); }
template<class T>inline void InputF(T& v) { cin >> v; }
inline void InputF(char& v) { while (isspace(v = gc())); }
inline void InputF(bool& v) { char c; InputF(c); v = c != '0'; }
inline void InputF(string& v) {
	v.clear(); char c; for (InputF(c); !isspace(c); c = gc())v += c;
}
inline void InputF(int& v) {
	bool neg = false; v = 0; char c; InputF(c);
	if (c == '-') { neg = true; c = gc(); }
	for (; isdigit(c); c = gc())v = v * 10 + (c - '0');
	if (neg)v = -v;
}
inline void InputF(long long& v) {
	bool neg = false; v = 0; char c; InputF(c);
	if (c == '-') { neg = true; c = gc(); }
	for (; isdigit(c); c = gc())v = v * 10 + (c - '0');
	if (neg)v = -v;
}
inline void InputF(double& v) {
	double dp = 1; bool neg = false, adp = false; v = 0; char c; InputF(c);
	if (c == '-') { neg = true; c = gc(); }
	for (; isdigit(c) || c == '.'; c = gc()) {
		if (c == '.')adp = true;
		else if (adp)v += (c - '0') * (dp *= 0.1);
		else v = v * 10 + (c - '0');
	}
	if (neg)v = -v;
}
template<class T, class U>inline void InputF(pair<T, U>& v) {
	InputF(v.first); InputF(v.second);
}
template<class T>inline void InputF(vector<T>& v) {
	for (auto& e : v)InputF(e);
}
template<class T>inline T InputF() { T v; InputF(v); return v; }
inline string GetLine() {
	string v; char c; for (InputF(c); c != '\n' && c != '\0'; c = gc())v += c; return v;
}
struct InputV {
	int n, m;
	InputV(int N) :n(N), m(0) {}
	InputV(pair<int, int> N) :n(N.first), m(N.second) {}
	template<class T>operator vector<T>() {
		vector<T> v(n); InputF(v); return v;
	}
	template<class T>operator vector<vector<T>>() {
		vector<vector<T>> v(n, vector<T>(m)); InputF(v); return v;
	}
};
struct Input {
	template<class T>operator T() { return InputF<T>(); }
	int operator--(int) { int v; InputF(v); v--; return v; }
	InputV operator[](int n) { return InputV(n); }
	InputV operator[](pair<int, int> n) { return InputV(n); }
	void operator()() {}
	template<class H, class...T>void operator()(H&& h, T&& ...t) {
		InputF(h); operator()(forward<T>(t)...);
	}
	template<class T>Input& operator,(T&& v) {
		InputF(v); return *this;
	}
}in;
#define input(T) InputF<T>()
#define ini input(int)
#define inl input(ll)
#define ins input(string)
#define inputs(T, ...) T __VA_ARGS__; in(__VA_ARGS__)
#define INT(...) inputs(int, __VA_ARGS__)
#define LL(...) inputs(ll, __VA_ARGS__)
#define STR(...) inputs(string, __VA_ARGS__)

// --- output --- //
struct BoolStr {
	const char* t, * f; BoolStr(const char* _t, const char* _f) :t(_t), f(_f) {}
}Yes("Yes", "No"), yes("yes", "no"), YES("YES", "NO"), Int("1", "0");
struct DivStr {
	const char* d, * l; DivStr(const char* _d, const char* _l) :d(_d), l(_l) {}
}spc(" ", "\n"), no_spc("", "\n"), end_line("\n", "\n"), comma(",", "\n"), no_endl(" ", "");
class Output {
	BoolStr B{ Yes }; DivStr D{ spc };
	void p(int v) {
		if (v < 0)putchar_unlocked('-'), v = -v;
		char b[10]; int i = 0;
		while (v)b[i++] = '0' + v % 10, v /= 10;
		if (!i)b[i++] = '0';
		while (i--)putchar_unlocked(b[i]);
	}
	void p(ll v) {
		if (v < 0)putchar_unlocked('-'), v = -v;
		char b[20]; int i = 0;
		while (v)b[i++] = '0' + v % 10, v /= 10;
		if (!i)b[i++] = '0';
		while (i--)putchar_unlocked(b[i]);
	}
	void p(bool v) { p(v ? B.t : B.f); }
	void p(char v) { putchar_unlocked(v); }
	void p(const char* v) { fwrite_unlocked(v, 1, strlen(v), stdout); }
	void p(double v) { printf("%.20f", v); }
	void p(ld v) { printf("%.20Lf", v); }
	template<class T> void p(const T& v) { cout << v; }
	template<class T, class U>void p(const pair<T, U>& v) { p(v.first); p(D.d); p(v.second); }
	template<class T>void p(const vector<T>& v) { rep(i, sz(v)) { if (i)p(D.d); p(v[i]); } }
	template<class T>void p(const vector<vector<T>>& v) { rep(i, sz(v)) { if (i)p(D.l); p(v[i]); } }
public:
	Output& operator()() { p(D.l); return *this; }
	template<class H>Output& operator()(H&& h) { p(h); p(D.l); return *this; }
	template<class H, class...T>Output& operator()(H&& h, T&& ...t) {
		p(h); p(D.d); return operator()(forward<T>(t)...);
	}
	template<class...T>void exit(T&& ...t) { operator()(forward<T>(t)...); std::exit(EXIT_SUCCESS); }
	Output& flush() { fflush_unlocked(stdout); return *this; }
	Output& set(const BoolStr& b) { B = b; return *this; }
	Output& set(const DivStr& d) { D = d; return *this; }
	Output& set(const char* t, const char* f) { B = BoolStr(t, f); return *this; }
}out;

// --- step --- //
template<class T>struct Step {
	class It {
		T a, b, c;
	public:
		constexpr It() : a(T()), b(T()), c(T()) {}
		constexpr It(T _b, T _c, T _s) : a(_b), b(_c), c(_s) {}
		constexpr It& operator++() { --b; a += c; return *this; }
		constexpr It operator++(int) { It tmp = *this; --b; a += c; return tmp; }
		constexpr const T& operator*()const { return a; }
		constexpr const T* operator->()const { return &a; }
		constexpr bool operator==(const It& i)const { return b == i.b; }
		constexpr bool operator!=(const It& i)const { return !(b == i.b); }
		constexpr T start()const { return a; }
		constexpr T count()const { return b; }
		constexpr T step()const { return c; }
	};
	constexpr Step(T b, T c, T s) : be(b, c, s) {}
	constexpr It begin()const { return be; }
	constexpr It end()const { return en; }
	constexpr T start()const { return be.start(); }
	constexpr T count()const { return be.count(); }
	constexpr T step()const { return be.step(); }
	constexpr T sum()const { return start() * count() + step() * (count() * (count() - 1) / 2); }
	operator vector<T>()const { return as_vector(); }
	vector<T> as_vector()const {
		vector<T> res; res.reserve(count()); each([&](T i) {res.push_back(i); }); return res;
	}
	template<class F>void each(const F& f)const { for (T i : *this)f(i); }
	template<class F>auto map(const F& f)const {
		vector<decay_t<result_of_t<F(T)>>> res; res.reserve(count());
		each([&](T i) {res.push_back(f(i)); }); return res;
	}
	template<class F>int count_if(const F& f)const {
		int res = 0; each([&](T i) {res += static_cast<bool>(f(i)); }); return res;
	}
	template<class F>vector<T> select(const F& f)const {
		vector<T> res; each([&](T i) {if (f(i))res.push_back(i); }); return res;
	}
	template<class F>auto max(const F& f)const {
		auto v = map(f); return *max_element(v.begin(), v.end());
	}
	template<class F>auto min(const F& f)const {
		auto v = map(f); return *min_element(v.begin(), v.end());
	}
	template<class F, class U = decay_t<result_of_t<F(T)>>>auto sum(const F& f)const {
		U res = 0; each([&](T i) {res += static_cast<U>(f(i)); }); return res;
	}
	using value_type = T;
	using iterator = It;
private:
	It be, en;
};
template<class T>inline constexpr auto step(T a) { return Step<T>(0, a, 1); }
template<class T>inline constexpr auto step(T a, T b) { return Step<T>(a, b - a, 1); }
template<class T>inline constexpr auto step(T a, T b, T c) { return Step<T>(a, a < b ? (b - a - 1) / c + 1 : 0, c); }

// --- functions --- //
inline namespace {
	template<class T>inline void Sort(T& a) { sort(all(a)); }
	template<class T>inline void RSort(T& a) { sort(rall(a)); }
	template<class T>inline T Sorted(T a) { Sort(a); return a; }
	template<class T>inline T RSorted(T a) { RSort(a); return a; }
	template<class T, class F>inline void Sort(T& a, const F& f) {
		sort(all(a), [&](const auto& x, const auto& y) {return f(x) < f(y); });
	}
	template<class T, class F>inline void RSort(T& a, const F& f) {
		sort(rall(a), [&](const auto& x, const auto& y) {return f(x) < f(y); });
	}
	template<class T>inline void Reverse(T& a) { reverse(all(a)); }
	template<class T>inline void Unique(T& a) { a.erase(unique(all(a)), a.end()); }
	template<class T>inline void Rotate(T& a, int left) { rotate(a.begin(), a.begin() + left, a.end()); }
	template<class T>inline T Reversed(T a) { Reverse(a); return a; }
	template<class T>inline T Uniqued(T a) { Unique(a); return a; }
	template<class T>inline T Rotated(T a, int left) { Rotate(a, left); return a; }
	template<class T>inline auto Max(const T& a) { return *max_element(all(a)); }
	template<class T>inline auto Min(const T& a) { return *min_element(all(a)); }
	template<class T>inline int MaxPos(const T& a) { return max_element(all(a)) - a.begin(); }
	template<class T>inline int MinPos(const T& a) { return min_element(all(a)) - a.begin(); }
	template<class T, class F>inline auto Max(const T& a, const F& f) {
		return *max_element(all(a), [&](const auto& x, const auto& y) {return f(x) < f(y); });
	}
	template<class T, class F>inline auto Min(const T& a, const F& f) {
		return *min_element(all(a), [&](const auto& x, const auto& y) {return f(x) < f(y); });
	}
	template<class T, class U>inline int Count(const T& a, const U& v) { return count(all(a), v); }
	template<class T, class F>inline int CountIf(const T& a, const F& f) { return count_if(all(a), f); }
	template<class T, class U>inline int Find(const T& a, const U& v) { return find(all(a), v) - a.begin(); }
	template<class T, class F>inline int FindIf(const T& a, const F& f) { return find_if(all(a), f) - a.begin(); }
	template<class T, class U = typename T::value_type>inline U Sum(const T& a) { return accumulate(all(a), U()); }
	template<class T, class F>inline auto Sum(const T& v, const F& f) {
		return accumulate(next(v.begin()), v.end(), f(v.front()), [&](auto a, auto b) {return a + f(b); });
	}
	template<class T, class U>inline int Lower(const T& a, const U& v) { return lower_bound(all(a), v) - a.begin(); }
	template<class T, class U>inline int Upper(const T& a, const U& v) { return upper_bound(all(a), v) - a.begin(); }
	template<class T, class F>inline void RemoveIf(T& a, const F& f) { a.erase(remove_if(all(a), f), a.end()); }
	template<class F>inline auto Vector(size_t size, const F& f) {
		vector<decay_t<result_of_t<F(size_t)>>> res(size); for (size_t i = 0; i < size; ++i)res[i] = f(i); return res;
	}
	template<class T>inline auto Grid(size_t h, size_t w, const T& v = T()) { return vector<vector<T>>(h, vector<T>(w, v)); }
	template<class T>inline auto Slice(const T& v, size_t i, size_t len) {
		return i < v.size() ? T(v.begin() + i, v.begin() + min(i + len, v.size())) : T();
	}
	template<class T, class F>inline auto Each(const T& v, const F& f) { for (auto& i : v)f(i); }
	template<class T, class F>inline auto Select(const T& v, const F& f) {
		T res; for (const auto& e : v)if (f(e))res.push_back(e); return res;
	}
	template<class T, class F>inline auto Map(const T& v, const F& f) {
		vector<decay_t<result_of_t<F(typename T::value_type)>>> res(v.size());
		size_t i = 0; for (const auto& e : v)res[i++] = f(e); return res;
	}
	template<class T, class F>inline auto MapIndex(const T& v, const F& f) {
		vector<decay_t<result_of_t<F(size_t, typename T::value_type)>>> res(v.size());
		size_t i = 0; for (auto it = v.begin(); it != v.end(); ++it, ++i)res[i] = f(i, *it); return res;
	}
	template<class T, class F>inline auto TrueIndex(const T& v, const F& f) {
		vector<size_t> res; for (size_t i = 0; i < v.size(); ++i)if (f(v[i]))res.push_back(i); return res;
	}
	inline string operator*(string s, size_t n) { string ret; for (size_t i = 0; i < n; ++i)ret += s; return ret; }
	template<class T>inline T Ceil(T n, T m) { return (n + m - 1) / m; }
	template<class T>inline T Ceil2(T n, T m) { return Ceil(n, m) * m; }
	template<class T>inline T Tri(T n) { return (n & 1) ? (n + 1) / 2 * n : n / 2 * (n + 1); }
	template<class T>inline T nC2(T n) { return (n & 1) ? (n - 1) / 2 * n : n / 2 * (n - 1); }
	template<class T>inline T Mid(const T& l, const T& r) { return l + (r - l) / 2; }
	inline int pop_count(int n) { return bitset<32>(n).count(); }
	inline int pop_count(ll n) { return bitset<64>(n).count(); }
	template<class T>inline bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
	template<class T>inline bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }
	template<class T>inline bool inRange(const T& v, const T& min, const T& max) { return min <= v && v < max; }
	template<class T = ll>inline T BIT(int b) { return T(1) << b; }
	template<class T>inline T Gcd(T n, T m) { return m ? Gcd(m, n % m) : n; }
	template<class T>inline T Lcm(T n, T m) { return n / Gcd(n, m) * m; }
	template<class T, class U = typename T::value_type>inline U Gcdv(const T& v) {
		return accumulate(next(v.begin()), v.end(), U(*v.begin()), Gcd<U>);
	}
	template<class T, class U = typename T::value_type>inline U Lcmv(const T& v) {
		return accumulate(next(v.begin()), v.end(), U(*v.begin()), Lcm<U>);
	}
	template<class T>inline T Pow(T a, T n) { T r = 1; while (n > 0) { if (n & 1)r *= a; a *= a; n /= 2; } return r; }
	template<class T>inline T Powmod(T a, T n, T m = MOD) {
		T r = 1; while (n > 0) { if (n & 1)r = r * a % m, n--; else a = a * a % m, n /= 2; }return r;
	}
}

// --- dump --- //
#if __has_include("dump.hpp")
#include "dump.hpp"
#else
#define dump(...) ((void)0)
#endif

// ---------------------------------------------------------------- //

bool check(const string& s) {
	if (s.back() == 'L')return false;
	VB flag(4);
	int d = 0;
	for (char c : s) {
		d = (d + (c == 'R' ? 1 : -1) + 4) % 4;
		if (c == 'R')flag[d] = true;
	}
	return Count(flag, true) == 4;
}

int main() {
	string s = in;
	int n = sz(s);

	VI pos_zero{ -1 };
	int d = 0;
	rep(i, n) {
		d = (d + (s[i] == 'R' ? 1 : -1) + 4) % 4;
		if (d == 0)pos_zero.push_back(i);
	}

	int ans = 0;
	rep(i, sz(pos_zero) - 1) {
		ans += check(s.substr(pos_zero[i] + 1, pos_zero[i + 1] - pos_zero[i]));
	}
	out(ans);
}
