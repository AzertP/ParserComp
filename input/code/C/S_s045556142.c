
int main(void){
    int h,w;
    while (1){
        scanf("%d %d",&h,&w);
        if(h==0 && w==0){
            break;
        }else{
            for (int i=0; i<h; i++){
                for (int j=0; j<w; j++){
                    printf("#");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}
