#include<stdio.h>

int main(void){
  int answer_t[256];
  int answer_h[256];
  int k;
  int a;
  for(k=0;k>=0;k++){
    int n,m;
    int i,j;
    int total_t = 0,total_h = 0;
    int ans_t = -1,ans_h;
    int tmp;
    int cmp;
    scanf("%d%d",&n,&m);
    if(n == 0 && m == 0)break;
    int taro[n];
    int hanako[m];
    for(i=0;i<n;i++){
      scanf("%d",&taro[i]);
      total_t += taro[i];
    }
    for(i=0;i<m;i++){
      scanf("%d",&hanako[i]);
      total_h += hanako[i];
    }
    cmp = total_t - total_h;
    if(cmp % 2 == 1 || cmp % 2 == -1){
      ans_t = -1;
    } 
    else if(cmp >= 0){
      for(i=0;i<m;i++){
	for(j=i;j<m;j++){
	  if(hanako[i]>hanako[j]){
	    tmp = hanako[i];
	    hanako[i] = hanako[j];
	    hanako[j] = tmp;
	  }
	}
      }
      for(i=0;i<m;i++){
	for(j=0;j<n;j++){
	  if(hanako[i] + cmp/2 == taro[j]){
	    ans_t = taro[j];
	    ans_h = hanako[i];
	  }
	  if(ans_t != -1)break;
	}
	if(ans_t != -1)break;
      }
    }
    else if(cmp < 0){
      cmp = cmp * -1;
      for(i=0;i<n;i++){
	for(j=i;j<n;j++){
	  if(taro[i] > taro[j]){
	    tmp = taro[i];
	    taro[i] = taro[j];
	    taro[j] = tmp;
	  }
	}
      }
      for(i=0;i<n;i++){
	for(j=0;j<m;j++){
	  if(taro[i] + cmp/2 == hanako[j]){
	    ans_t = taro[i];
	    ans_h = hanako[j];
	  }
	  if(ans_t != -1)break;
	}
	if(ans_t != -1)break;
      }
    }
    answer_t[k] = ans_t;
    answer_h[k] = ans_h;
  }
  for(a=0;a<k;a++){
    if(answer_t[a] == -1){
      printf("-1\n");
    }
    else{
      printf("%d %d\n",answer_t[a],answer_h[a]);
    }
  }
  return 0;
}
  