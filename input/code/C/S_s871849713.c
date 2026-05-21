#include<stdio.h>

int main(){
    int count = 0;
    long int sum = 0;

    scanf("%d", &count);

    for(int i = 1; i <= count ; i ++){
        if((i % 3 != 0) && (i % 5 != 0)){
            sum += i;
        }
    }

    printf("%ld\n", sum);

    return 0;
}