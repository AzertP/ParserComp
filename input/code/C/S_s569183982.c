
int main(void)
{
    int N, K;
    int b;
    int tower;
    int use;
    
    scanf("%d%d", &N, &K);

    tower = 1;  // 
    use = 1;    // 
    b = 1;      // 
    while ((N - use) >= b){
        if (use <= K * b){
            tower++;
            use += b;
        }
        else {
            b++;
        }
    }
    printf("%d\n", tower);
    
    return (0);
}

