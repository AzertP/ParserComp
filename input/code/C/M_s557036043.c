typedef int64_t ll;
typedef uint64_t ull;
int acs(const void *a, const void *b){return *(int*)a - *(int*)b;} /* 1,2,3,4.. */
int des(const void *a, const void *b){return *(int*)b - *(int*)a;} /* 8,7,6,5.. */


int p[20];
int main(void)
{
    int n;
    scanf("%d",&n);

    for(int i=0;i<n;i++)
    {
        scanf("%d",&(p[i]));
    }
    int ans = 0;
    for(int i=1;i<n-1;i++)
    {
        int pt[3] = {p[i-1],p[i],p[i+1]};
        qsort(pt,3,sizeof(int),acs);
        if(pt[1]==p[i]) ans++;
    }
    printf("%d\n",ans);
    return 0;
}
