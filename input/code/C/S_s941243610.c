//http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ITP1_5_A&lang=ja

    
    int main(void){
        int w,h,i,j;
        while(1){
            scanf("%d %d",&w,&h);
            if(w == 0 && h == 0){
                break;
            }else{
                for(i=0;i<w;i++){
                    for(j=0;j<h;j++){
                        printf("#");
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }

        return 0;
    }
