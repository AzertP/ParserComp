#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int compare_string(const void *left, const void *right) {
    char *left_char = (char *)left;
    char *right_char = (char *)right;
    
    return strcmp(left_char,right_char);
    
}


int main(){
    int i,n,count=1;
    char names[200001][11];
    scanf("%d",&n);
    for(i = 0; i < n; i++) scanf("%s",names[i]);
    qsort(names, n, sizeof *names, compare_string);
    
    for(i = 1; i < n; i++){
        if(strcmp(names[i-1],names[i]) != 0) count++;
    }
//    for(i = 0; i < n; i++) printf("%s\n",names[i]);
    printf("%d\n",count);
    return 0;
}
