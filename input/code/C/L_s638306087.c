#include<stdio.h>
#include<string.h>
#include<math.h>
#include<stdlib.h>
#include<limits.h>

#define rep(i,begin,end) for(int i=begin; i<end; i++)
#define revrep(i,begin,end) for(int i=begin; i>end; i--)
#define lld long long int

int main(){
    int n;
    scanf("%d", &n);
    char s[n+1];
    scanf("%s", s);
    int buf[n+1];
    int f = 0;
    rep(i, 0, 4){
        if(i == 0){
            buf[0] = 1;
            buf[1] = 1;
        }else if(i == 1){
            buf[0] = 1;
            buf[1] = 0;
        }else if(i == 2){
            buf[0] = 0;
            buf[1] = 0;
        }else{
            buf[0] = 0;
            buf[1] = 1;
        }
        rep(j, 1, n){
            if(s[j] == 'o'){
                if(buf[j]){
                    buf[j+1] = buf[j-1];
                }else{
                    buf[j+1] = (buf[j-1] + 1) % 2;
                }
            }else{
                if(buf[j]){
                    buf[j+1] = (buf[j-1] + 1) % 2;
                }else{
                    buf[j+1] = buf[j-1];
                }
            }
        }
        if(buf[0] == buf[n]){
            if(s[0] == 'o'){
                if(buf[0]){
                    if(buf[n-1] == buf[1]){
                        f = 1;
                        break;
                    }
                }else{
                    if(buf[n-1] != buf[1]){
                        f = 1;
                        break;
                    }
                }
            }else{
                if(buf[0]){
                    if(buf[n-1] != buf[1]){
                        f = 1;
                        break;
                    }
                }else{
                    if(buf[n-1] == buf[1]){
                        f = 1;
                        break;
                    }
                }
            }
        }
    }
    if(f){
        rep(i, 0, n){
            printf("%c", buf[i] ? 'S' : 'W');
        }
    }else{
        printf("-1");
    }

    return 0;
}