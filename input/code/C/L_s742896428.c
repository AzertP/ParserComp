

//quicksort http://www.math.u-ryukyu.ac.jp/~suga/C/2004/13/node6.html
//http://d.hatena.ne.jp/lagos_on/20090517/1242586942
void qsort(int list1[],int list2[], int left, int right) {
    int l = left;
    int r = right;
    int pivot = list1[(left + right) / 2]; // 
    int temp;

    while (1) {
        while (list1[l] < pivot) l++;
        while (list1[r] > pivot) r--;

        if (l > r) break;

        // pivot
        temp = list1[l];
        list1[l] = list1[r];
        list1[r] = temp;
        
        temp = list2[l];
        list2[l] = list2[r];
        list2[r] = temp;


        l++, r--;
    };

    // pivot
    if (left < r)  qsort(list1,list2 ,left, r);
    if (l < right) qsort(list1,list2, l, right);
}

void quicksort(long long int list[],int list2[], int left, int right)
{
    int i, last;
    int temp;
    if (left >= right)
        return;

    last = left;
    for (i=left+1; i <= right; i++){
        if (list[i] < list[left] ){
            last++;
            temp=list[last];
            list[last]=list[i];
            list[i]=temp;
            temp=list2[last];
            list2[last]=list2[i];
            list2[i]=temp;
        }
    }

    temp=list[left];
    list[left]=list[last];
    list[last]=temp;

    temp=list2[left];
    list2[left]=list2[last];
    list2[last]=temp;

    quicksort(list,list2, left, last-1);
    quicksort(list,list2, last+1, right);
}
int main(void){
    int N,M;
    scanf("%d %d",&N,&M);
    int A[N];
    int B[N];
    for(int i=0;i<N;i++){
        scanf("%d %d",&A[i],&B[i]);
    }
    long long int money = 0;
    qsort(A,B,0,N-1);
/*
    printf("------------\n");
    for(int i=0;i<N;i++){
        printf("%d %d\n",A[i],B[i]);
    }
//*/
    for(int i=0;i<N;i++){
        if(M>B[i]){
            money += (long long int)A[i]*B[i];
            M -= B[i];
            //printf("money = %lld ,M=%d\n",money,M);
        }else{
            money += (long long int)A[i]*M;
            M -= M;
            break;
        }
    }

    printf("%lld",money);
    return 0;
}
