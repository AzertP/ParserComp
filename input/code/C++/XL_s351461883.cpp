
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <sstream>
#include <string>
#include <functional>
#include <queue>
#include <deque>
#include <stack>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <fstream>
#include <iterator>
#include <random>
#include <chrono>

 
#define forr(i,start,count) for (int i = (start); i < (start)+(count); ++i)
#define set_map_includes(set, elt) (set.find((elt)) != set.end())
#define readint(i) int i; cin >> i
#define readll(i) ll i; cin >> i
#define readdouble(i) double i; cin >> i
#define readstring(s) string s; cin >> s
 
typedef long long ll;
 
using namespace std;

ll modd = 1000*1000*1000+7;


template<class T>
class depth_first_search_iterator {
  public:
    stack<pair<T,T>> todo;
    function<bool(T)> notvisited;
    function<void(T)> mark_visited;
    function<set<T>(T)> neighbors;

    depth_first_search_iterator() {}

    depth_first_search_iterator(T start, function<bool(T)> notv, function<void(T)> mark_v, function<set<T>(T)> neigh) :
      notvisited(notv), mark_visited(mark_v), neighbors(neigh), end_(false), current(start), prev(start) {
          todo.push(make_pair(start, start)); operator++();
    }

    void operator++() {
        end_ = true;
        while (!todo.empty()) {
            current = todo.top().first; prev = todo.top().second; todo.pop();
            if (notvisited(current)) {
                mark_visited(current);
                for (const T x : neighbors(current)) {  todo.push(make_pair(x, current));   }
                end_ = false;
                break;
            }
        }
    }

    T operator*() {     return current;    }

    T previous() {  return prev;  }

    bool end() {  return end_;  }

  private:
    T current; // contains current node
    T prev;    // contains previous node
    bool end_;
};

template<class T>
class breadth_first_search_iterator {
  public:
    queue<pair<T,T>> todo;
    function<bool(T)> notvisited;
    function<void(T)> mark_visited;
    function<set<T>(T)> neighbors;

    breadth_first_search_iterator() {}

    breadth_first_search_iterator(T start, function<bool(T)> notv, function<void(T)> mark_v, function<set<T>(T)> neigh) :
      notvisited(notv), mark_visited(mark_v), neighbors(neigh), end_(false), current(start), prev(start) {
          todo.push(make_pair(start, start)); operator++();
    }

    void operator++() {
        end_ = true;
        while (!todo.empty()) {
            current = todo.front().first; prev = todo.front().second; todo.pop();
            if (notvisited(current)) {
                mark_visited(current);
                for (const T x : neighbors(current)) {  todo.push(make_pair(x, current));   }
                end_ = false;
                break;
            }
        }
    }

    T operator*() {     return current;    }

    T previous() {  return prev;  }

    bool end() {  return end_;  }

  private:
    T current;   // current node
    T prev;      // previous node
    bool end_;
};

template<class T>
class DirectedGraph {
    // allows only single connection between two vertices
    public:
      vector<T> vertices;
      vector<set<int>> neighbors;
      depth_first_search_iterator<int> dfs_iterator;
      breadth_first_search_iterator<int> bfs_iterator;
      vector<bool> visited;

      DirectedGraph() {}

      DirectedGraph(int n, T default_val = 0) {  for (int i = 0; i < n; ++i) {  AddVertex(default_val);  }   }

      bool EdgeExists(int i_from, int j_to) {
          return (neighbors[i_from].find(j_to) != neighbors[i_from].end());
      }

      void AddVertex(T val) {
          vertices.push_back(val);
          neighbors.push_back(set<int>());      }

      void AddEdge(int i_from, int j_to) {
          if (!EdgeExists(i_from, j_to)) {     neighbors[i_from].insert(j_to);      }
      }

      T& operator[](int i) {      return vertices[i];      }

      void RemoveEdge(int i_from, int j_to) {
          if (EdgeExists(i_from, j_to)) {
          neighbors[i_from].erase(find(neighbors[i_from].begin(), neighbors[i_from].end(), j_to));      } }

      DirectedGraph<T> Transpose() {
          DirectedGraph<T> ret;
          for (auto x : vertices) {  ret.AddVertex(x);   }
          for (int i_from = 0; i_from < vertices.size(); ++i_from) {
              for (int j_to : neighbors[i_from]) {
                  ret.AddEdge(j_to, i_from);
              }
          }
          return ret;      }

      int vertices_count() {   return vertices.size();  }

      void dfs_init(int start) {
          visited = vector<bool>(vertices.size(), false);
          dfs_iterator = depth_first_search_iterator<int>(start, [this](int x){ return !visited[x]; }, [this](int x) { visited[x] = true;  },
            [this](int x) {  return neighbors[x];  } );
      }

      void bfs_init(int start) {
          visited = vector<bool>(vertices.size(), false);
          bfs_iterator = breadth_first_search_iterator<int>(start, [this](int x){ return !visited[x]; }, [this](int x) { visited[x] = true;  },
            [this](int x) {  return neighbors[x];  } );
      }

      void print_() {
          forr(i, 0, vertices.size()) {
              for (auto x : neighbors[i]) {
                  cout << i << "->" << x << endl;
              }
          }
      }

};


int main()   {


    ios_base::sync_with_stdio(false);

    cout.precision(17);
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> rand_gen(0, modd);   // rand_gen(rng) gets the rand no

//    auto start = chrono::steady_clock::now()

//    readint(test_cases);
    int test_cases = 1;
    forr(t, 1, test_cases) {
        readstring(s);
        int ret = 0;
        if (s=="SSS") {ret = 0;}
        if (s=="SSR") {ret = 1;}
        if (s=="SRS") {ret = 1;}
        if (s=="SRR") {ret = 2;}
        if (s=="RSS") {ret = 1;}
        if (s=="RSR") {ret = 1;}
        if (s=="RRS") {ret = 2;}
        if (s=="RRR") {ret = 3;}
        cout << ret << endl;
    }

//    auto end = chrono::steady_clock::now();
//    cerr << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    return 0;
}