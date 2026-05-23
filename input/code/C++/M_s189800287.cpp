
long table[1000010];
long h[100010];
long N,K;

long solve(long goal, long K)
{
    std::vector<long> temporary;
    if(table[goal]==LONG_MAX){
        for(long i=1; i<=K;i++){
            if(goal-i>=0){
                temporary.push_back(solve(goal-i,K)+abs(h[goal]-h[goal-i]));
            }
        }
        std::sort(temporary.begin(),temporary.end());
        table[goal]=temporary[0];
    }
    return table[goal];
}


int main()
{
    std::cin >> N >> K;
    for(long i=0; i<N; i++){
        std::cin >> h[i];
        table[i]=LONG_MAX;
    }
    table[0]=0;
    table[1]=abs(h[1]-h[0]);
    std::cout << solve(N-1,K) << std::endl;
    return 0;
}
