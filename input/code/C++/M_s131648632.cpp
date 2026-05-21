/*  -*- coding: utf-8 -*-
 *
 * e.cc: E - Red and Green Apples
 */

#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<iostream>
#include<string>
#include<vector>
#include<map>
#include<set>
#include<stack>
#include<list>
#include<queue>
#include<deque>
#include<algorithm>
#include<numeric>
#include<utility>
#include<complex>
#include<functional>
 
using namespace std;

/* constant */

const int MAX_N = 100000;
const int MAX_N2 = MAX_N * 2;

/* typedef */

typedef long long ll;

/* global variables */

int ps[MAX_N], qs[MAX_N], rs[MAX_N];
int ss[MAX_N2];
ll sss[MAX_N2 + 1], rss[MAX_N + 1];

/* subroutines */

/* main */

int main() {
  int x, y, a, b, c;
  scanf("%d%d%d%d%d", &x, &y, &a, &b, &c);

  for (int i = 0; i < a; i++) scanf("%d", ps + i);
  for (int i = 0; i < b; i++) scanf("%d", qs + i);
  for (int i = 0; i < c; i++) scanf("%d", rs + i);

  sort(ps, ps + a);
  sort(qs, qs + b);
  sort(rs, rs + c);

  int n = 0, pk = a, qk = b;
  for (int i = 0; i < x; i++) ss[n++] = ps[--pk];
  for (int i = 0; i < y; i++) ss[n++] = qs[--qk];
  sort(ss, ss + n);

  for (int i = 0; i < n; i++) sss[i + 1] = sss[i] + ss[i];
  for (int i = 0; i < c; i++) rss[i + 1] = rss[i] + rs[i];

  ll maxsum = sss[n];
  int l = min(n, c);
  for (int i = 0; i <= l; i++) {
    ll sum = sss[n] - sss[i] + (rss[c] - rss[c - i]);
    if (maxsum < sum) maxsum = sum;
  }

  printf("%lld\n", maxsum);
  return 0;
}
