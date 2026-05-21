// AOJ Volume 1 Problem 0163 Ohajiki Game

#include <stdio.h>


int main(void)
{
    int i;
    int j;
    int a[25];
    int ohajiki;
    int jiro_turn;
    
    while (1){
        scanf("%d", &i);
        if (i == 0){
            break;
        }

        for (j = 0; j < i; j++){
            scanf("%d", &a[j]);
        }
        
        jiro_turn = 0;
        ohajiki = 32;
        while (1){
            // êYÌ^[
            ohajiki -= ((ohajiki - 1) % 5);
            
            // ¨Í¶«Ì\¦
            printf("%d\n", ohajiki);

            // ¨Í¶«ªÈ­ÈÁ½çI¹
            if (ohajiki == 0){
                break;
            }
            
            // YÌ^[
            ohajiki -= a[jiro_turn];
            jiro_turn = ((jiro_turn + 1) % i);

            // ¨Í¶«ªÈ­ÈÁ½çI¹
            if (ohajiki <= 0){
                printf("0\n");
                break;
            }

            // ¨Í¶«Ì\¦
            printf("%d\n", ohajiki);
        }
    }
    
    return (0);
}