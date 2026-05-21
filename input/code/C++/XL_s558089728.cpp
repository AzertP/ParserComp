#ifndef _GLIBCXX_NO_ASSERT
#include <cassert>
#endif
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#if __cplusplus >= 201103L
#include <ccomplex>
#include <cfenv>
#include <cinttypes>
#include <cstdalign>
#include <cstdbool>
#include <cstdint>
#include <ctgmath>
#include <cwchar>
#include <cwctype>
#endif
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>
#if __cplusplus >= 201103L
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <forward_list>
#include <future>
#include <initializer_list>
#include <mutex>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <system_error>
#include <thread>
#include <tuple>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#endif
#define y0 qvya13579
#define y1 qvyb24680
#define j0 qvja13579
#define j1 qvjb24680
#define next qvne13579xt
#define prev qvpr13579ev
#define INF 1000000007
#define MOD 1000000007
#define PI acos(-1.0)
#define endl "\n"
#define IOS cin.tie(0);ios::sync_with_stdio(false)
#define M_P make_pair
#define PU_B push_back
#define PU_F push_front
#define PO_B pop_back
#define PO_F pop_front
#define U_B upper_bound
#define L_B lower_bound
#define B_S binary_search
#define PR_Q priority_queue
#define FIR first
#define SEC second
#if __cplusplus < 201103L
#define stoi(argument_string) atoi((argument_string).c_str())
#define stoll(argument_string) atoll((argument_string).c_str())
#endif
#define REP(i,n) for(int i=0;i<(int)(n);++i)
#define REP_R(i,n) for(int i=((int)(n)-1);i>=0;--i)
#define FOR(i,m,n) for(int i=((int)(m));i<(int)(n);++i)
#define FOR_R(i,m,n) for(int i=((int)(m)-1);i>=(int)(n);--i)
#define ALL(v) (v).begin(),(v).end()
#define RALL(v) (v).rbegin(),(v).rend()
#define SIZ(x) ((int)(x).size())
#define CIN(x) cin>>(x)
#define CIN2(x,y) cin>>(x)>>(y)
#define CIN3(x,y,z) cin>>(x)>>(y)>>(z)
#define CIN4(x,y,z,w) cin>>(x)>>(y)>>(z)>>(w)
#define CIN5(x,y,z,w,u) cin>>(x)>>(y)>>(z)>>(w)>>(u)
#define SCAND(x) scanf("%d",&(x))
#define SCAND2(x,y) scanf("%d%d",&(x),&(y))
#define SCAND3(x,y,z) scanf("%d%d%d",&(x),&(y),&(z))
#define SCAND4(x,y,z,w) scanf("%d%d%d%d",&(x),&(y),&(z),&(w))
#define SCAND5(x,y,z,w,u) scanf("%d%d%d%d%d",&(x),&(y),&(z),&(w),&(u))
#define SCANLLD(x) scanf("%lld",&(x))
#define SCANLLD2(x,y) scanf("%lld%lld",&(x),&(y))
#define SCANLLD3(x,y,z) scanf("%lld%lld%lld",&(x),&(y),&(z))
#define SCANLLD4(x,y,z,w) scanf("%lld%lld%lld%lld",&(x),&(y),&(z),&(w))
#define SCANLLD5(x,y,z,w,u) scanf("%lld%lld%lld%lld%lld",&(x),&(y),&(z),&(w),&(u))
#define I64DSCAN(x) scanf("%I64d",&(x))
#define I64DSCAN2(x,y) scanf("%I64d%I64d",&(x),&(y))
#define I64DSCAN3(x,y,z) scanf("%I64d%I64d%I64d",&(x),&(y),&(z))
#define I64DSCAN4(x,y,z,w) scanf("%I64d%I64d%I64d%I64d",&(x),&(y),&(z),&(w))
#define I64DSCAN5(x,y,z,w,u) scanf("%I64d%I64d%I64d%I64d%I64d",&(x),&(y),&(z),&(w),&(u))
#define PRINTD(x) printf("%d\n",(x))
#define PRINTLLD(x) printf("%lld\n",(x))
#define PRINTI64D(x) printf("%I64d\n",(x))
#define DEBUG(argument) cerr<<(#argument)<<" : "<<(argument)<<"\n"
typedef long long int lli;
using namespace std;


bool compare_by_2nd(pair<int,int> a, pair<int,int> b)
{
  if( a.second != b.second )
    {
      return a.second < b.second;
    }
  else
    {
      return a.first < b.first;
    }
  
}

int ctoi(char c)
{
  if( c >= '0' and c <= '9' )
    {
      return (int)(c-'0');
    }

  return -1;
}

int alphabet_pos(char c) 
{
  if( c >= 'a' and c <= 'z' )
    {
      return (int)(c-'a');
    }

  return -1;
}


int alphabet_pos_capital(char c)
{
  if( c >= 'A' and c <= 'Z' )
    {
      return (int)(c-'A');
    }

  return -1;
}


vector<string> split(string str, char ch)
{
  int first = 0;
  int last = str.find_first_of(ch);
  
  if(last == string::npos)
    {
      last = SIZ(str);
    }

  vector<string> result;

  while( first < SIZ(str) )
    {
      string Ssubstr(str, first, last - first);
      result.push_back(Ssubstr);
      first = last + 1;
      last = str.find_first_of(ch, first);

      if(last == string::npos)
	{
	  last = SIZ(str);
	}
    }
  
  return result;
}



int gcd( int a , int b ) // assuming a,b >= 1
{
  if( a < b )
    {
      swap(a,b);
    }

  if( a == 0 )
    {
      return b;
    }
  
  if( a % b == 0 )
    {
      return b;    
    }

  return gcd( b , a % b );
  
}

long long gcd( long long a , long long b ) // assuming a,b >= 1
{
  if( a < b )
    {
      swap(a,b);
    }

  if( a == 0LL )
    {
      return b;
    }
  
  if( a % b == 0 )
    {
      return b;
    }

  return gcd( b , a % b );
  
}

int lcm( int a , int b ) // assuming a,b >= 1
{
  return  a * b / gcd( a , b );
  
}

long long lcm( long long a , long long b ) // assuming a,b >= 1
{
  return  a * b / gcd( a , b );
  
}

long long pow_fast( long long x, long long n_power , long long modulus )
{
  
  if( n_power == 0 )
    {
      return 1;
    }
  
  if( n_power % 2 == 0)
    {
      return pow_fast( x * x % modulus , n_power / 2 , modulus );  
    }

  
  return x * pow_fast( x , n_power - 1 , modulus ) % modulus;
}


struct CombinationTable
{
  vector<vector<long long> > val;

  CombinationTable( int size ) : val( size+1 , vector<long long>( size+1 ) ) //constructor
  {
    for( int i = 0 ; i <= size ; ++ i ) // note that 0 <= i <= size
      {

	for( int j = 0 ; j <= i ; ++ j )
	  {
	    
	    if( j == 0 or j == i )
	      {
		val[i][j] = 1LL;
	      }
	    else
	      {
		val[i][j] = val[i-1][j-1] + val[i-1][j];
	      }

	  }
	
	
      }
    
  }

  
};


void print_vector(vector<int>& h)
{
  int L =  h.size();
  
  for(int i = 0; i < L; ++ i)
    {
      printf("%d",h[i]);

      if( i != L-1 )
	{
	  printf(" ");
	}
      else
	{
	  printf("\n");
	}
    }    
}

void print_vector(vector<long long>& h)
{
  int L =  h.size();
  
  for(int i = 0; i < L; ++ i)
    {
      printf("%lld",h[i]);

      if( i != L-1 )
	{
	  printf(" ");
	}
      else
	{
	  printf("\n");
	}
    }    
}



void print_matrix2D(vector<vector<int> >& h)
{
  int Ly =  h.size();
  int Lx = h[0].size();
        
  for(int i = 0; i < Ly; ++ i)
    {
      for(int j = 0; j < Lx; ++ j)
        {
	  printf("%d",h[i][j]);

	  if( j != Lx-1 )
	    {
	      printf(" ");
	    }
	  else
	    {
	      printf("\n");
	    }	  
        }
    }    
}

void print_matrix2D(vector<vector<long long> >& h)
{
  int Ly =  h.size();
  int Lx = h[0].size();
        
  for(int i = 0; i < Ly; ++ i)
    {
      for(int j = 0; j < Lx; ++ j)
        {
	  printf("%lld",h[i][j]);

	  if( j != Lx-1 )
	    {
	      printf(" ");
	    }
	  else
	    {
	      printf("\n");
	    }
        }
    }
}

void print_matrix2D(vector<string>& h)
{
  int Ly =  h.size();
  int Lx = h[0].size();
        
  for(int i = 0; i < Ly; ++ i)
    {
      for(int j = 0; j < Lx; ++ j)
        {
	  printf("%c",h[i][j]);
        }
      printf("\n");
    }    
}

void print_binary(int val, int num_digit = 31, bool verbose = false)
{
  for(int k = num_digit-1; k >= 0; -- k)
    {
      if( verbose )
	{
	  printf("%d",(val>>k)&1);
	}
      else
	{
	  cerr << ((val>>k)&1);
	}    
    }
  
  if( verbose )
    {
      printf("\n");
    }
  else
    {
      cerr << "\n";
    }
}

void print_binary(long long val, int num_digit = 63, bool verbose = false)
{
  for(int k = num_digit-1; k >= 0; -- k)
    {
      if( verbose )
	{
	  printf("%lld",((val>>k)&1));
	}
      else
	{
	  cerr << ((val>>k)&1);
	}    
    }
  
  if( verbose )
    {
      printf("\n");
    }
  else
    {
      cerr << "\n";
    }
}



struct UnionFind //size-based
{
  vector<int> parent, treesize;
  
  UnionFind( int size ) : parent( size ) , treesize( size , 1 ) //constructor
  {
    for( int i = 0 ; i < size ; ++ i )
      {
	parent[i] = i;
      }
  }

  int root( int x )
  {
    if( parent[x] == x )
      {
	return x;
      }
    
    return parent[x] = root(parent[x]);
  }

  void unite( int x, int y )
  {
    x = root(x); 
    y = root(y);
    
    if( x == y )
      {
	return; 
      }

    if( treesize[x] < treesize[y] )
      {
	parent[x] = y;
	treesize[y] += treesize[x];
      }
    else
      {
	parent[y] = x;
	treesize[x] += treesize[y];	
      }
    
  }

  bool sametree( int x, int y )
  {
    return root(x) == root(y);
  }

  int gettreesize( int x )
  {
    return treesize[root(x)];
  }
  
};

template< typename Type_value >
struct SegmentTree //Range Minimum Query (RMQ)
{

private:
  int n;
  vector<Type_value> node;
  Type_value identity_element_segmenttree;
  
public:
  SegmentTree( vector<Type_value> v, Type_value identity_element_st ) //constructor
  {
    int sz = v.size();
    identity_element_segmenttree = identity_element_st;
    n = 1;
    while(n < sz)
      {
	n <<= 1;
      }
    
    node.resize(2*n-1, identity_element_segmenttree);

    for( int i = 0 ; i < sz ; ++ i )
      {
	node[i+n-1] = v[i];
      }
    
    for( int i = n-2 ; i >= 0 ; -- i )
      {
	node[i] = min(node[2*i+1], node[2*i+2]);
      } 
  }
 
  void update(int x, Type_value val)
  {
    x += (n - 1);
    node[x] = val;
    
    while( x > 0 )
      {
	x = (x - 1) / 2;
	node[x] = min(node[2*x+1], node[2*x+2]);
      }
  }
 
  Type_value getmin( int a, int b, int k = 0, int l = 0, int r = -1 ) //getting minimum value in [a,b)
  {
    // k : index of the referred node
    // [l,r) : range covered by the k-th node 
    
    if( r < 0 )
      {
	r = n;
      } 
    if( r <= a or b <= l )
      {
	return identity_element_segmenttree;
      } 
    if( a <= l and r <= b )
      {
	return node[k];
      }
    
    Type_value vl = getmin(a, b, 2*k+1, l, (l+r)/2);
    Type_value vr = getmin(a, b, 2*k+2, (l+r)/2, r);
    return min(vl, vr);
  }
  
};


template< typename Type_value >
struct BinaryIndexedTree //Range Sum Query (RSQ), 0-indexed
{
  
private:
  int size_;
  vector< Type_value > data;

public:
  BinaryIndexedTree(int sz, Type_value identity_element_binaryindexedtree = 0.0 ) // constructor
  {
    size_ = sz;
    data.resize(sz+1,identity_element_binaryindexedtree);
  }

  Type_value sum(int i) //sum within [0,i)
  {
    if( i <= 0 )
      {
	return (Type_value) 0.0;
      }

    if( i > size_ )
      {
	i = size_;
      }
    
    Type_value sm = 0.0;
    while( i > 0 )
      {
	sm += data[i];
	i -= i & -i;
      }
    
    return sm;
  }

  void add(int i, Type_value x)
  {
    if( i < 0 or i >= size_ )
      {
	return;
      }
    ++ i;
    while( i <= size_ )
      {
	data[i] += x;
	i += i & -i;
      }
  }
};

struct RollingHash
{

  /*
    10 primes near 1e9(for other mod(s)):
    {
    1000000007,
    1000000009,
    1000000021,
    1000000033,
    1000000087,
    1000000093,
    1000000097,
    1000000103,
    1000000123,
    1000000181
    }

    10 primes near 1e5(for other base(s)):
    {100003,
    100019,
    100043,
    100049,
    100057,
    100069,
    100103,
    100109,
    100129,
    100151}
  */
  
private:
  vector<unsigned long long> hash;
  vector<unsigned long long> base_pow;
  unsigned long long base, modulus;

  const unsigned long long MODULUS_DEFAULT = (1ULL << 61) - 1;
  const unsigned long long MASK30 = (1ULL << 30) - 1;
  const unsigned long long MASK31 = (1ULL << 31) - 1;
  const unsigned long long BASE_MIN = 1e7;

  unsigned long long Modulus_2pow61m1( unsigned long long val )
  {
    val = (val & MODULUS_DEFAULT) + (val >> 61);

    if( MODULUS_DEFAULT < val )
      {
	val -= MODULUS_DEFAULT;
      }

    return val;
  }
  

  unsigned long long Multiple_2pow61m1( unsigned long long a, unsigned long long b )
  {
    unsigned long long au = a >> 31;
    unsigned long long ad = a & MASK31;
    unsigned long long bu = b >> 31;
    unsigned long long bd = b & MASK31;

    unsigned long long mid = ad * bu + au * bd;
    unsigned long long midu = mid >> 30;
    unsigned long long midd = mid & MASK30;

    return Modulus_2pow61m1( ( (au * bu) << 1 ) + midu + (midd << 31) + ad * bd );
  }
 
  void initialize(string& S)
  {
    int N = S.size();
    vector<int> s(N);
    for(int i = 0; i < N; ++i)
      {
	s[i] = S[i];
      }
    
    initialize(s);
  }
 
  void initialize(vector<int>& S)
  {
    hash.resize(S.size()+1);
    base_pow.resize(S.size()+1);
    hash[0] = 0;
    base_pow[0] = 1;

    if( modulus == MODULUS_DEFAULT )
      {	
	for(int i = 1; i <= S.size(); ++i)
	  {
	    hash[i] = Modulus_2pow61m1( Multiple_2pow61m1(hash[i-1],base) + S[i-1] );
	    base_pow[i] = Multiple_2pow61m1(base_pow[i-1],base);
	  }
      }
    else
      {
	for(int i = 1; i <= S.size(); ++i)
	  {
	    hash[i] = ( hash[i-1] * base + S[i-1] ) % modulus;
	    base_pow[i] = ( base_pow[i-1] * base ) % modulus;
	  }
      }
  }

public:
  RollingHash(string S = "", unsigned long long base_ = 0, unsigned long long modulus_ = 0)
  {
    if( 0 < modulus_ )
      {
	modulus = modulus_;
      }
    else
      {
	modulus = MODULUS_DEFAULT;
      }

    if( 0 < base_ )
      {
	base = base_;
      }
    else
      {
	mt19937_64 mt64(static_cast<unsigned int>(time(nullptr)));
	uniform_int_distribution<unsigned long long> rand_uniform(BASE_MIN,modulus-BASE_MIN);
	base = rand_uniform(mt64);
      }

    if( S.size() > 0 )
      {
	initialize(S);
      }
  }
  
  // 0-indexed, [a, b)
  unsigned long long between(int a, int b)
  {
    if( modulus == MODULUS_DEFAULT )
      {
	return Modulus_2pow61m1( modulus + hash[b] - Multiple_2pow61m1(hash[a],base_pow[b-a]) );
      }
    
    return ( modulus + hash[b] - ( hash[a] * base_pow[b-a] ) % modulus ) % modulus;
  }
};


/*------------------ the end of the template -----------------------*/









signed main()
{
  IOS; /* making cin faster */
  int N;
  SCAND(N);
  vector<int> a(N);
  vector<int> b(N);
  int x = 0;
  REP(i,N)
    {
      SCAND(a[i]);
      x ^= a[i];
      b[i] = a[i] ^ (a[i] - 1);
    }

  vector<bool> used(N,false);

  int cnt = 0;
  REP_R(k,30)
    {
      
      bool flag_used = false;
      REP(i,N)
	{
	  if( not used[i] and (b[i] >> k)&1 )
	    {
	      if( not flag_used and (x >> k)&1 )
		{
		  ++ cnt;
		  x ^= b[i];
		}
	      
	      used[i] = true;
	      flag_used = true;
	    }
	}
      
    }


  if( x == 0 )
    {
      PRINTD(cnt);
    }
  else
    {
      PRINTD(-1);
    }
  
  

}
