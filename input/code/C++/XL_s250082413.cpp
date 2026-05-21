#define NDEBUG
#include <vector>
#include <set>
#include <algorithm>
#include <tuple>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace mm {
    inline void getCpuClock(unsigned long long & t) {
        __asm__ __volatile__ ("rdtsc" : "=a"(*(unsigned int*)&t), "=d"(((unsigned int*)&t)[1]));
    }

    inline double getNativeTime() {
        timeval tv;
        gettimeofday(&tv, 0);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    unsigned long long g_initCpuClock;
    unsigned long long g_reserveUpdateCpuClock;
    double g_initNativeTime;
    double g_secPerClock;
    double g_doneTime;
    inline void initTime() {
        g_initNativeTime = getNativeTime();
        getCpuClock(g_initCpuClock);
        g_secPerClock = 0.00000000025;
        g_reserveUpdateCpuClock = 10000000;
        g_doneTime = 0;
    }

    inline double getTime() {
        unsigned long long now;
        getCpuClock(now);
        now -= g_initCpuClock;
        if(g_reserveUpdateCpuClock < now) {
            double nowTime = getNativeTime() - g_initNativeTime;
            g_secPerClock = nowTime / now;
            g_reserveUpdateCpuClock = now + (unsigned long long)(0.05 * now / nowTime);
        }
        return g_doneTime = std::fmax(g_doneTime, g_secPerClock * now);
    }
}

namespace mm {
    inline unsigned int asm_mul_hi(unsigned int x, unsigned int y) {
        __asm__ __volatile__("mul %%edx" : "+a"(x), "+d"(y));
        return y;
    }
    unsigned long long g_rand49_state = 0x8a5cd789635d2dffULL;
    inline int mrand49() {
        g_rand49_state *= 6364136223846793005ULL;
        g_rand49_state += 1442695040888963407ULL;
        unsigned int ret = ((g_rand49_state>>18)^g_rand49_state) >> 27;
        unsigned int rot = (g_rand49_state>>59);
        return (ret>>rot) | (ret<<-rot);
    }
    inline long long mmrand49() {
        int v = mrand49();
        return ((long long)v << 32) | mrand49();
    }
    inline int lrand49() {
        return mrand49() & 0x7FFFFFFF;
    }
    inline int lrand49(int r) {
        assert(1<=r);
        return asm_mul_hi(mrand49(), r);
        //return ((unsigned long long)(unsigned int)mrand49() * r)>>32;
    }

    inline double drand49() {
        return ((unsigned int)mrand49() + 0.5) * (1.0/4294967296.0);
    }

    inline void srand49(int seed) {
        g_rand49_state = seed + 1442695040888963407ULL;
    }
    inline void srand49() {
        int clk;
        __asm__ __volatile__ ("rdtsc" : "=a"(*(unsigned int*)&clk) : : "%rdx");
        srand49(clk);
    }
}

using namespace std;
using namespace mm;

#define arraysizeof(a) (sizeof(a)/sizeof(a[0]))
#define rep(i, n) for(int i=0; i<(int)(n); ++i)
constexpr int I = 26;
constexpr double limit = 1.95;
int D;
int c[I];
int s[365][I];
int t[365];
int cnt;
double rnd[8192];
double Tr;
double T;
set<int> days[I];

bool accept(int dSc) {
    //if(0<=dSc) return true;
    return T * rnd[cnt&(arraysizeof(rnd)-1)] < dSc;
}

int pena(int dd) {
    return (dd * (dd+1)) >> 1;
}

int score(int i, int d) {
    auto it = days[i].lower_bound(d);
    int nd = D;
    if(it!=days[i].end()) {
        nd = *it;
    }
    int pd = 0;
    if(it!=days[i].begin()) {
        --it;
        pd = *it;
    }
    return s[d][i] - (pena(nd-d) + pena(d-pd) - pena(nd-pd)) * c[i];
}

void mutateA() {
    int d = lrand49(D);
    int i1 = t[d];
    int i2 = lrand49(I-1);
    if(i1<=i2) ++i2;
    days[i1].erase(d);
    int dSc = score(i2, d) - score(i1, d);
    if(accept(dSc)) {
        days[i2].insert(d);
        t[d] = i2;
    }
    else {
        days[i1].insert(d);
    }
}

constexpr int ddMax = 14;

void mutateB() {
    int dd = lrand49(ddMax) + 1;
    int d1 = lrand49(D-dd);
    int d2 = d1 + dd;
    int i1 = t[d1];
    int i2 = t[d2];
    if(i1==i2) return;
    days[i1].erase(d1);
    days[i2].erase(d2);
    int dSc = score(i1, d2) + score(i2, d1) - score(i1, d1) - score(i2, d2);
    if(accept(dSc)) {
        days[i1].insert(d2);
        days[i2].insert(d1);
        t[d1] = i2;
        t[d2] = i1;
    }
    else {
        days[i1].insert(d1);
        days[i2].insert(d2);
    }
}

void mutateC() {
    int dd = lrand49(ddMax) + 1;
    int dd2 = lrand49(ddMax-1) + 1;
    if(dd<=dd2) {
        ++dd2;
    }
    else {
        swap(dd, dd2);
    }
    int d1 = lrand49(D-dd2);
    int d2 = d1 + dd;
    int d3 = d1 + dd2;
    int i1 = t[d1];
    int i2 = t[d2];
    int i3 = t[d3];
    if(i1==i2 || i2==i3 || i1==i3) {
        return;
    }
    days[i1].erase(d1);
    days[i2].erase(d2);
    days[i3].erase(d3);
    int sc11 = score(i1, d1);
    int sc12 = score(i1, d2);
    int sc13 = score(i1, d3);
    int sc21 = score(i2, d1);
    int sc22 = score(i2, d2);
    int sc23 = score(i2, d3);
    int sc31 = score(i3, d1);
    int sc32 = score(i3, d2);
    int sc33 = score(i3, d3);
    int sc123 = sc11 + sc22 + sc33;
    int sc132 = sc11 + sc23 + sc32;
    int sc213 = sc12 + sc21 + sc33;
    int sc231 = sc12 + sc23 + sc31;
    int sc312 = sc13 + sc21 + sc32;
    int sc321 = sc13 + sc22 + sc31;
    int max_sc = max(max(max(sc123, sc132), max(sc213, sc231)), max(sc312, sc321));
    sc123 -= max_sc;
    sc132 -= max_sc;
    sc213 -= max_sc;
    sc231 -= max_sc;
    sc312 -= max_sc;
    sc321 -= max_sc;
    double p123 = exp(sc123 / T);
    double p132 = exp(sc132 / T);
    double p213 = exp(sc213 / T);
    double p231 = exp(sc231 / T);
    double p312 = exp(sc312 / T);
    double p321 = exp(sc321 / T);
    double sum = p123 + p132 + p213 + p231 + p312 + p321;
    double r = drand49() * sum;
    if(r<p123) {
        days[i1].insert(d1);
        days[i2].insert(d2);
        days[i3].insert(d3);
        return;
    }
    p132 += p123;
    if(r<p132) {
        days[i1].insert(d1);
        days[i2].insert(d3);
        days[i3].insert(d2);
        t[d3] = i2;
        t[d2] = i3;
        return;
    }
    p213 += p132;
    if(r<p213) {
        days[i1].insert(d2);
        days[i2].insert(d1);
        days[i3].insert(d3);
        t[d2] = i1;
        t[d1] = i2;
        return;
    }
    p231 += p213;
    if(r<p231) {
        days[i1].insert(d2);
        days[i2].insert(d3);
        days[i3].insert(d1);
        t[d2] = i1;
        t[d3] = i2;
        t[d1] = i3;
        return;
    }
    p312 += p231;
    if(r<p312) {
        days[i1].insert(d3);
        days[i2].insert(d1);
        days[i3].insert(d2);
        t[d3] = i1;
        t[d1] = i2;
        t[d2] = i3;
        return;
    }
    days[i1].insert(d3);
    days[i2].insert(d2);
    days[i3].insert(d1);
    t[d3] = i1;
    t[d1] = i3;
}

vector<tuple<int, int, int, int, int> > mVec;
vector<double> dVec;
void mutateD() {
    int dd = lrand49(ddMax) + 1;
    int dd2 = lrand49(ddMax-1) + 1;
    if(dd<=dd2) {
        ++dd2;
    }
    else {
        swap(dd, dd2);
    }
    int dd3 = lrand49(ddMax-2) + 1;
    if(dd<=dd3) {
        ++dd3;
        if(dd2<=dd3) {
            ++dd3;
        }
        else {
            swap(dd2, dd3);
        }
    }
    else {
        swap(dd, dd3);
        swap(dd2, dd3);
    }
    int d1 = lrand49(D-dd3);
    int d2 = d1 + dd;
    int d3 = d1 + dd2;
    int d4 = d1 + dd3;
    int i1 = t[d1];
    int i2 = t[d2];
    int i3 = t[d3];
    int i4 = t[d4];
    if(i1==i2 || i1==i3 || i1==i4 || i2==i3 || i2==i4 || i3==i4) {
        return;
    }
    int iList[4] = {i1, i2, i3, i4};
    int dList[4] = {d1, d2, d3, d4};
    rep(i, 4) days[iList[i]].erase(dList[i]);
    int scList[4][4];
    rep(i, 4) rep(j, 4) scList[i][j] = score(iList[i], dList[j]);
    mVec.clear();
    dVec.clear();
    rep(a1, 4) {
        int sc1 = scList[0][a1];
        rep(a2, 4) if(a1!=a2) {
            int sc2 = sc1 + scList[1][a2];
            rep(a3, 4) if(a1!=a3 && a2!=a3) {
                int sc3 = sc2 + scList[2][a3];
                rep(a4, 4) if(a1!=a4 && a2!=a4 && a3!=a4) {
                    int sc4 = sc3 + scList[3][a4];
                    mVec.emplace_back(sc4, a1, a2, a3, a4);
                    break;
                }
            }
        }
    }
    sort(mVec.begin(), mVec.end());
    int max_value = get<0>(mVec.back());
    double sum = 0;
    rep(i, mVec.size()) {
        double e = exp(get<0>(mVec[i]) - max_value);
        sum += e;
        dVec.emplace_back(e);
    }
    double r = drand49() * sum;
    rep(i, mVec.size()) {
        double e = dVec[i];
        if(i==mVec.size()-1 || r < e) {
            days[i1].insert(dList[get<1>(mVec[i])]);
            days[i2].insert(dList[get<2>(mVec[i])]);
            days[i3].insert(dList[get<3>(mVec[i])]);
            days[i4].insert(dList[get<4>(mVec[i])]);
            t[dList[get<1>(mVec[i])]] = i1;
            t[dList[get<2>(mVec[i])]] = i2;
            t[dList[get<3>(mVec[i])]] = i3;
            t[dList[get<4>(mVec[i])]] = i4;
            return;
        }
        dVec[i+1] += e;
    }
}

int main() {
    initTime();
    srand49();
    scanf("%d", &D);
    rep(i, arraysizeof(rnd)) {
        double r = drand49();
        rnd[i] = log(r) - log(1-r);
    }
    rep(i, I) scanf("%d", &c[i]);
    rep(d, D) rep(i, I) scanf("%d", &s[d][i]);
    rep(d, D) t[d] = d % I;
    rep(d, D) days[t[d]].insert(d);
    cnt = 0;
    int cnt2 = 0;
    while(true) {
        double now = getTime();
        if(limit<=now) {
            break;
        }
        if(!((++cnt2) & 15)) {
            constexpr double T0 = 1500.0;
            double progress = now / limit;
            double remain = 1 - progress;
            double remain05 = sqrt(remain);
            constexpr double rate05 = 0.3;
            Tr = (1-rate05) * remain + rate05 * remain05;
            T = T0 * Tr;
        }
        mutateA(); ++cnt;
        mutateB(); ++cnt;
        mutateC(); ++cnt;
    }
    //fprintf(stderr, "cnt = %d\n", cnt2);
    rep(d, D) printf("%d\n", t[d]+1);
}
