


int main(void){
    int x;
    scanf("%d",&x);
    for(int i=1;;i++){
        if(x*i%360==0){printf("%d",i);return 0;}
    }
}
