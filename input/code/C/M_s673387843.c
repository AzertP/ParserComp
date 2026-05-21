//Twitter@KonoLv1 2020年春からエンジニア見習いデビュー！
 
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
 
int main(void){
    int i,j;            //  ループ文に使用
    int x,y,z;
    int count1,count2,count;   //  カウント用に使用
    double k;
    double n;
    int boxa[1024] = {};
    int boxb[1024] = {};
    char s[200001] = {};        //  文字列
    char s2[200001] = {};        //  文字列
    int a,b,c,h,w;
    a = b = c = i = 0;
    int mode = 0;
    count = count1 = count2 = 0;
 
    //  入力処理
    
    scanf("%lf",&n);
    for (i = 1; i <= n; i++){
        if (i % 2 == 1){
            count++;
        }

    }
         k = count / n;
    //  計算処理
    
 
 
    //  出力処理
 
    printf("%2.10lf",k);
 
}