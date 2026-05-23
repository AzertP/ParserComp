/* 164b.c */

int main(void){
    int a,b,c,d=0;
    int tak,aok=0;
    scanf("%d %d %d %d",&a,&b,&c,&d);
    aok=c-b;
    tak=a-d;
    if(aok<=0){
        printf("Yes");
        return 0;
    }else if(tak<=0){
        printf("No");
        return 0;
    }
    for(int i=0; i<100;i++){
        aok=aok-b;
        tak=tak-d;
        if(aok<=0){
            printf("Yes");
            return 0;
        }else if(tak<=0){
            printf("No");
            return 0;
        }
    }
}
