void input_array(long how_data,long *data);
void output_array(long how_data,long *data);
void input_array2(long first , long second , long data[][2]);
void format_array(long how_data ,long *data,long what);
long get_random(long min, long max);
long factorial(long n);
long fibonacci(long n);
int qsort_09(const int *sys1 , const int *sys2);
int qsort_90(const int *sys1 , const int *sys2);
long sel_max(long a , long b);
long sel_min(long a , long b);
long array_max(long how_data,long *data);
long array_min(long how_data,long *data);
long array_search(long how_data,long *data,long what);
long can_DP(long how_data,long *data,long how_can,bool *can);
long array_sum(long how_data,long *data);
long Leven_dist(char *now , char *target);
long get_digit(long target);
long ncr(long n , long r);
long npr(long n , long r);
long nhr(long n , long r);
int fifw(long add,long *data);
long fifr(long *data);
void GRAPH_input(void);
void GRAPH_dfs(long now);

//
long fifo[1000]={0};//fifo[0]
long loop1,loop2,loop3,loop4,loop5,i_temp;
long g_ans=0; //answer

typedef struct{
    long how_vertex;
    long how_edge;
    long edge_data[1000][2];
    bool aleady_searched_vertex[500];
    bool all_searched;
} _GD;//GraphData

_GD GRAPH={0,0,{{0}},{0},0};

char c_temp;


int main(void){
    char s[10000];
    scanf("%s",s);
    In(target_x);In(target_y);
    int next_syori=0,now_x=0,now_y=0;
    /*
        txy
        xy
        xyYes
        No
        
        F
        F
    */
    
    int front_F=0;
    while(s[next_syori]=='F' && next_syori<strlen(s)){
        front_F++;
        next_syori++;
    }
    target_x-=front_F;
    
    /*
        F
        Y
        target+-
    */
    
    long how_x=0,how_y=0;
    bool next_xy=0; //0=x 1=y
    for(loop1=next_syori;loop1<strlen(s);loop1++){
        if(s[loop1]=='T'){
            next_xy=(!next_xy);
            if(next_xy==1)how_y++;
            if(next_xy==0)how_x++;
        }
    }
    //printf("%d %d\n",how_x,how_y);
    
    long data_x[how_x],data_y[how_y];
    format_array(how_x,data_x,0);
    format_array(how_y,data_y,0);
    long ct_x=-1,ct_y=-1;
    next_xy=0;
    for(loop1=next_syori;loop1<strlen(s);loop1++){
        if(s[loop1]=='T'){
            next_xy=(!next_xy);
            if (next_xy==0)ct_x++;
            if (next_xy==1)ct_y++;
        }else{
            if(next_xy==0){
                data_x[ct_x]++;
            }else{
                data_y[ct_y]++;
            }
        }
    }
    QS90(how_x,data_x);
    QS90(how_y,data_y);
    //printf("TGT X=%ld Y=%ld\n",target_x,target_y);
    //x
    
    //x
    //printf("List-x\n");
    //for(loop1=0;loop1<how_x;loop1++){
        //printf("%ld\n",data_x[loop1]);
    //}
    //printf("/\n");
    
    bool x_can=0;
    for(loop1=0;loop1<how_x;loop1++){
        if(now_x>target_x){
            now_x-=data_x[loop1];
        }else{
            now_x+=data_x[loop1];
        }
        //printf("%d\n",now_x);
    }
    if (now_x==target_x){
        x_can=1;
    }
    
    //y
    bool y_can=0;
    for(loop1=0;loop1<how_y;loop1++){
        if(now_y>target_y){
            now_y-=data_y[loop1];
        }else{
            now_y+=data_y[loop1];
        }
    }
    if (now_y==target_y){
        y_can=1;
    }
    
    
    
    if(x_can==1 && y_can==1){
        printf("Yes");
    }else{
        printf("No");
    }
    
    END;
    
}
 

 
void input_array(long how_data,long *data){
    long loop;
    for(loop=0;loop<how_data;loop++){
        scanf("%ld",&data[loop]);
    }
    return ;
}
void output_array(long how_data,long *data){
    long loop;
    for(loop=0;loop<how_data-1;loop++){
        printf("%ld ",data[loop]);
    }
    return ;
}
void input_array2(long first,long second,long data[][2]){
    long loopA,loopB;
    for(loopA=0;loopA<first;loopA++){
        for(loopB=0;loopB<second;loopB++){
            scanf("%ld",&data[loopA][loopB]);
        }
    }
    return ;
}
void format_array(long how_data ,long *data,long what){
    long loopA;
    for(loopA=0;loopA<how_data;loopA++){
        data[loopA]=what;
    }
    return ;
}
long get_random(long min, long max){   //1
    //srand((unsigned int)time(0));   //
    rand();rand();rand();rand();    //
    return (rand()%(max+1-min))+min;
}
long factorial(long n){//n!MOD10^9+7
    unsigned long long int ret=1;
    for(long i=1;i<=n;i++)ret=(ret*i)%1000000007;
    return (long)ret;
}
int qsort_09(const int *sys1 , const int *sys2){ //
    //qsort(data,how_data,sizeof(long),(int (*)(const void *,const void *))qsort_09);
    if(*sys1<*sys2){
        return -1;
    }else if(*sys1==*sys2){
        return 0;
    }else{
        return 1;
    }
}
int qsort_90(const int *sys1 , const int *sys2){ //
    //qsort(data,how_data,sizeof(long),(int (*)(const void *,const void *))qsort_90);
    if(*sys1<*sys2){
        return 1;
    }else if(*sys1==*sys2){
        return 0;
    }else{
        return -1;
    }
}
long fibonacci(long n){
    switch(n){
        case 0:return 0;
        case 1:return 1;
        default :return fibonacci(n-1)+fibonacci(n-2);
    }
}
long sel_max(long a,long b){
    if(a>b)return a;
    return b;
}
long sel_min(long a,long b){
    if(a>b)return b;
    return a;
}
long can_DP(long how_data,long *data,long how_can,bool *can){//Typical DP Contest A
    //datacan0 or 1
    //
    long loopA,loopB;
    long ret=0;
    for(loopA=0;loopA<how_can;loopA++){//
        can[loopA]=0;
    }
    can[0]=1;//
    for(loopA=0;loopA<how_data;loopA++){
        for(loopB=how_can-1;loopB>=0;loopB--){
            if(can[loopB]==1 && loopB+data[loopA]<how_can){
                can[loopB+data[loopA]]=1;
            }
        }
    }
    for(loopA=0;loopA<how_can;loopA++){
        if(can[loopA]==1){
            ret++;
        }
    }
    return ret;
}
long array_max(long how_data,long *data){
    long loop;
    long ret=data[0];
    for(loop=0;loop<how_data;loop++){
        if(ret<data[loop])ret=data[loop];
    }
    return ret;
}
long array_min(long how_data,long *data){
    long loop;
    long ret=data[0];
    for(loop=0;loop<how_data;loop++){
        if(ret>data[loop])ret=data[loop];
    }
    return ret;
}
long array_sum(long how_data,long *data){
    long ret=0;
    long loop;
    for(loop=0;loop<how_data;loop++){
        ret+=data[loop];
    }
    return ret;
}
long array_search(long how_data,long *data,long what){
    long loop;
    for(loop=0;loop<how_data;loop++){
        if(data[loop]==what){
            return loop;
        }
    }
    return -1;
}
long Leven_dist(char *now , char *target){
    long loopA,loopB;
    //
    // 
    // http://nw.tsuda.ac.jp/class/algoB/c13.html  ()
    // http://d.hatena.ne.jp/ohnishiakira/20090809/1249845529  ()
    long len_now=strlen(now)+1;
    long len_target=strlen(target)+1;
    long d[len_now][len_target]; //
    for(loopA=0;loopA<len_now;loopA++) d[loopA][0]=loopA;
    for(loopA=0;loopA<len_target;loopA++) d[0][loopA]=loopA;
    for(loopA=1;loopA<len_now;loopA++){
        for(loopB=1;loopB<len_target;loopB++){
            long cost=(now[loopA-1]==target[loopB-1] ? 0:1);
            d[loopA][loopB]=sel_min(sel_min(d[loopA-1][loopB]+1,d[loopA][loopB-1]+1),d[loopA-1][loopB-1]+cost);
        }
    }
    return d[len_now-1][len_target-1];
}
long get_digit(long target){
    return (long)(log10(target)+1);
}
long ncr(long n , long r){
    //
    long loopA,loopB;
    long pascal[100][102]={{0}};
    pascal[1][0]=1;
    pascal[1][1]=1;
    for(loopA=2;loopA<100;loopA++){
        pascal[loopA][0]=1;
        for(loopB=1;loopB<loopA;loopB++){
            pascal[loopA][loopB]=pascal[loopA-1][loopB-1]+pascal[loopA-1][loopB];
        }
        pascal[loopA][loopA]=1;
    }
    return pascal[n][r];
}
long npr(long n, long r){
    return ncr(n,r)*factorial(r);
}
long nhr(long n , long r){
    return ncr(n+r-1,r);
}
int fifw(long add,long *data){
    data[data[0]+1]=add;
    data[0]++;
    return data[0]; //
}
long fifr(long *data){
    long ret=data[1];
    data[0]--;
    for(int i=1;i<999;i++){
        data[i]=data[i+1];
    }
    return ret;
}
void GRAPH_input(void){
    scanf("%ld %ld\n",&GRAPH.how_vertex,&GRAPH.how_edge);
    for(long loopA=0;loopA<GRAPH.how_edge;loopA++){
        scanf("%ld %ld\n",&GRAPH.edge_data[loopA][0],&GRAPH.edge_data[loopA][1]);
    }
    return;
}

void GRAPH_dfs(long now){
    GRAPH.aleady_searched_vertex[now]=1;
    for(int loopA=0;loopA<GRAPH.how_edge;loopA++){
        if(GRAPH.edge_data[loopA][0]==now && GRAPH.aleady_searched_vertex[GRAPH.edge_data[loopA][1]]==0){
            GRAPH_dfs(GRAPH.edge_data[loopA][1]);
        }
        if(GRAPH.edge_data[loopA][1]==now && GRAPH.aleady_searched_vertex[GRAPH.edge_data[loopA][0]]==0){
            GRAPH_dfs(GRAPH.edge_data[loopA][0]);
        }
    }
    return ;
}


