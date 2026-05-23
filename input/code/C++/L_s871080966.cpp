using namespace std;

typedef long double cood;

const cood EPS=1e-8;

typedef complex<cood> point;
const point IU=point((cood)0,(cood)1);
namespace std{
  bool operator<(point P1,point P2){
    return abs(P1.real()-P2.real())>EPS?P1.real()<P2.real():P1.imag()<P2.imag();
  }
  bool operator>(point P1,point P2){
    return abs(P1.real()-P2.real())>EPS?P1.real()>P2.real():P1.imag()>P2.imag();
  } 
}
cood ep(point P1,point P2){return imag((conj(P1)*P2));}
cood ip(point P1,point P2){return real((conj(P1)*P2));}
int sign(cood s){return abs(s)<=EPS?0:s>0?1:-1;}
int ccw(point P0,point P1,point P2){return sign(ep(P1-P0,P2-P0));}
cood len(point P){return sqrt(norm(P));}
point uni(point P){return P/len(P);}
cood dis(point P1,point P2){return len(P1-P2);}
point cross_point(point P1,point P2,point P3,point P4){
  cood r=ep(P3-P1,P2-P1)/ep(P4-P3,P2-P1);
  return P3-r*(P4-P3);
}
point proj(point P,point P1,point P2){
  return P1+ip(P2-P1,P-P1)/norm(P2-P1)*(P2-P1);
}
point ref(point P,point P1,point P2){return (cood)2*proj(P,P1,P2)-P;}
// 1:P-P1-P2 2:P1-P-P2 3:P1-P2-P
int line_pos(point P,point P1,point P2){
  cood p=ip(P2-P1,P-P1);
  return p<-EPS?1:p<=norm(P2-P1)+EPS?2:3;
}
bool on_seg(point P,point P1,point P2){
  return ccw(P,P1,P2)==0&&line_pos(P,P1,P2)==2;
}
bool crossLS(point P0,point P1,point P2,point P3){
  return ccw(P0,P1,P2)*ccw(P0,P1,P3)<=0;//inclusive
}
bool crossSS(point P0,point P1,point P2,point P3){
  if(ccw(P0,P1,P2)==0&&ccw(P0,P1,P3)==0){
    int p1=line_pos(P2,P0,P1),p2=line_pos(P3,P0,P1);
    return !((p1==1&&p2==1)||(p1==3&&p2==3));
  }
  return crossLS(P0,P1,P2,P3)&&crossLS(P2,P3,P0,P1);
}
cood disPL(point P,point P1,point P2){return abs(ep(P-P1,P2-P1))/len(P2-P1);}
cood disPS(point P,point P1,point P2){
  point F=proj(P,P1,P2);
  return line_pos(F,P1,P2)==2?dis(P,F):min(dis(P,P1),dis(P,P2));
}
cood disSS(point P0,point P1,point P2,point P3){
  return crossSS(P0,P1,P2,P3)?(cood)0:min(min(disPS(P0,P2,P3),disPS(P1,P2,P3)),min(disPS(P2,P0,P1),disPS(P3,P0,P1)));
}
cood costh(cood a,cood b,cood c){//cosine theorem
  return (a*a+b*b-c*c)/(2*a*b);
}
const cood Pi=(cood)2*asin((cood)1);
cood ang(point P1,point P2,point P3){//ang P2P1P3
  return arg((P3-P1)/(P2-P1));
}

const int MAXN=110;

void convex_hull(point *P,int n,int *H,int &N)
{
  int v1[MAXN],v2[MAXN];
  v1[0]=v2[0]=0;
  int m1=1,m2=1;
  for(int i=1;i<n;i++){
    while(m1>=2&&ccw(P[v1[m1-2]],P[v1[m1-1]],P[i])<=EPS){
      m1--;
    }
    v1[m1]=i;
    m1++;
    while(m2>=2&&ccw(P[v2[m2-2]],P[v2[m2-1]],P[i])>=-EPS){
      m2--;
    }
    v2[m2]=i;
    m2++;
  }
  N=m1+m2-2;
  for(int i=0;i<m1-1;i++){
    H[i]=v1[i];
  }
  for(int i=0;i<m2-1;i++){
    H[m1-1+i]=v2[m2-1-i];
  }
}

int main()
{
  int n;
  scanf("%d",&n);
  pair<point,int> P2[110];
  for(int i=0;i<n;i++){
    cood x,y;
    scanf("%Lf%Lf",&x,&y);
    P2[i]=make_pair(point(x,y),i);
  }
  sort(P2,P2+n);
  point P[110];
  int rv[110];
  for(int i=0;i<n;i++){
    P[i]=P2[i].first;
    rv[i]=P2[i].second;
  }
  int H[110];
  int N=0;
  convex_hull(P,n,H,N);
  cood ans[110]={0};
  for(int i=0;i<N;i++){
    ans[rv[H[i]]]=Pi-ang(P[H[i]],P[H[(i+1)%N]],P[H[(i+N-1)%N]]);
  }
  for(int i=0;i<n;i++){
    printf("%.20Lf\n",ans[i]/(2*Pi));
  }
  return 0;
}
