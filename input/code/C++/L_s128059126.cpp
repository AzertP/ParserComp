#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
using namespace std;
typedef long long ll;
#define rep(i,n) for(ll i=0;i<n;++i)
#define rep1(i,n) for(ll i=1;i<=n;++i)
const int sup=101;
char map[sup][sup];
int main(){
  int a,b;
  cin>>a>>b;
  int tmp_a=a;
  int tmp_b=b;
  if(a>b){
    a=tmp_b;
    b=tmp_a;
  }
  int small=a;
  small--;
  int div=small/100;
  int rest=small%100;
  char buf='.';
  rep1(i,100){
    rep1(j,100){
      if(i==sup/2)buf='#';//最初は.途中から#
      map[i][j]=buf;
      if(i==sup/2&&j%2==1)map[i][j]='.';//50段目の奇数が白
      //      cout<<map[i][j];
    }
    //    cout<<endl;
  }
  int small2=small*2;//真ん中をあれする
  for(int i=sup/2+1;i<sup/2+5+div*2;i++){
    for(int j=1;j<=100;j++){
      if((i+j)%2==1)buf='.';
      else buf='#';
      if(small2==0)goto kk;
      map[i][j]=buf;
      small2--;
    }
  }
 kk:
  //aは.でbは#
  a-=small;
  b-=small;
  a--;
  b--;
  int tmp=1;
  if(a==0){
    rep1(i,25){
      rep1(j,100){
	if(b==0)goto FINE;
	if(i%2==1&&j%2==1){
	  map[i][j]='#';
	  b--;
	}
	//	if(b==0)goto FINE;
      }
    }
  }
  if(b==0){
    for(ll i=70;i<=100;i++){
      rep1(j,25){
	if(a==0)goto FINE;
	if(i%2==1&&j%2==1){
	  map[i][j]='.';
	  a--;
	}
	//	if(a==0)goto FINE;
      }
    }
  }
 FINE:
  cout<<"100 100"<<endl;
  rep1(i,100){
    rep1(j,100){
      if(tmp_a>tmp_b){
	if(map[i][j]=='.')map[i][j]='#';
	else map[i][j]='.';
      }
      cout<<map[i][j];
    }
    cout<<endl;
  }
  //  cout<<small<<endl;
  return 0;
}
