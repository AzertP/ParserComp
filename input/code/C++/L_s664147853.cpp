#include <iostream>
#include <vector>
#include <cstdio>
#include <sstream>
#include <map>
#include <string>
#include <algorithm>
#include <queue>
#include <cmath>
#include <functional>
#include <set>
#include <ctime>
#include <random>
#include <chrono>
#include <cassert>
using namespace std;

namespace {
  using Integer = long long; //__int128;
  template<class T> istream& operator >> (istream& is, vector<T>& vec){for(T& val: vec) is >> val; return is;}
  template<class T> istream& operator ,  (istream& is, T& val){ return is >> val;}
  template<class T> ostream& operator << (ostream& os, const vector<T>& vec){for(int i=0; i<vec.size(); i++) os << vec[i] << (i==vec.size()-1?"":" "); return os;}
  template<class T> ostream& operator ,  (ostream& os, const T& val){ return os << " " << val;}

  template<class H> void print(const H& head){ cout << head; }
  template<class H, class ... T> void print(const H& head, const T& ... tail){ cout << head << " "; print(tail...); }
  template<class ... T> void println(const T& ... values){ print(values...); cout << endl; }

  template<class H> void eprint(const H& head){ cerr << head; }
  template<class H, class ... T> void eprint(const H& head, const T& ... tail){ cerr << head << " "; eprint(tail...); }
  template<class ... T> void eprintln(const T& ... values){ eprint(values...); cerr << endl; }

  class range{
    long long start_, end_, step_;
   public:
    struct range_iterator{
      long long val, step_;
      range_iterator(long long v, long long step) : val(v), step_(step) {}
      long long operator * (){return val;}
      void operator ++ (){val += step_;}
      bool operator != (range_iterator& x){return step_ > 0 ? val < x.val : val > x.val;}
    };
    range(long long len) : start_(0), end_(len), step_(1) {}
    range(long long start, long long end) : start_(start), end_(end), step_(1) {}
    range(long long start, long long end, long long step) : start_(start), end_(end), step_(step) {}
    range_iterator begin(){ return range_iterator(start_, step_); }
    range_iterator   end(){ return range_iterator(  end_, step_); }
  };

  string operator "" _s (const char* str, size_t size){ return move(string(str)); }
  constexpr Integer my_pow(Integer x, Integer k, Integer z=1){return k==0 ? z : k==1 ? z*x : (k&1) ? my_pow(x*x,k>>1,z*x) : my_pow(x*x,k>>1,z);}
  constexpr Integer my_pow_mod(Integer x, Integer k, Integer M, Integer z=1){return k==0 ? z%M : k==1 ? z*x%M : (k&1) ? my_pow_mod(x*x%M,k>>1,M,z*x%M) : my_pow_mod(x*x%M,k>>1,M,z);}
  constexpr unsigned long long operator "" _ten (unsigned long long value){ return my_pow(10,value); }

  inline int k_bit(Integer x, int k){return (x>>k)&1;} //0-indexed

  mt19937 mt(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count());

  template<class T> string join(const vector<T>& v, const string& sep){
    stringstream ss; for(int i=0; i<v.size(); i++){ if(i>0) ss << sep; ss << v[i]; } return ss.str();
  }

  string operator * (string s, int k){ string ret; while(k){ if(k&1) ret += s; s += s; k >>= 1; } return ret; }

}

constexpr long long mod = 9_ten + 7;


vector<long long> Eratosthenes(long long N){
  vector<bool> v(N+1, true);
  v[0] = v[1] = false;
  long long sqN = sqrt(N);
  for(int i=2; i<=sqN; i++){
    if(v[i] == false) continue;
    for(long long j=i*i; j<=N; j+=i){
      v[j] = false;
    }
  }
  vector<long long> Prime;
  for(long long i=2; i<=N; i++){
    if(v[i]==true) Prime.push_back(i);
  }
  return Prime;
}

#include <unordered_map>


int main(){
  int n;
  cin >> n;
  vector<long long> v(n);
  for(int i=0; i<n; i++){
    scanf("%lld", &v[i]);
  }
  auto P = Eratosthenes(100000);

  unordered_map<long long, long long> square;
  for(auto p : P){
    square[p*p] = p;
  }

  vector<long long> cube;
  for(long long i=2; i*i*i <= 10_ten; i++){
    cube.push_back(i*i*i);
  }

  long long ans = 0;

  unordered_map<long long, long long> cnt;

  for(auto& x : v){
    for(auto c : cube){
      while(x%c == 0) x/=c;
    }
    if(x == 1) ans = 1;
    else cnt[x]++;
  }

  for(auto& x : cnt){
    if(x.second == 0){
      continue;
    }

    long long tmp = x.first;
    long long target = 1;
    for(auto p : P){
      if(p*p*p > 10_ten) break;
      if(tmp == 1) break;
      if(tmp%p != 0) continue;
      target *= tmp%(p*p) == 0 ? p : p*p;
      tmp /= tmp%(p*p) == 0 ? p*p : p;
    }

    if(tmp != 1){
      target *= square.count(tmp) ? square[tmp] : tmp*tmp;
    }

    if(cnt.count(target)){
      ans += max(x.second, cnt[target]);
      cnt[target] = 0;
    }else{
      ans += x.second;
    }
  }
  println(ans);

  return 0;
}
