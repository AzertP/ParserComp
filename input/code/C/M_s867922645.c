// AOJ Volume 1 Problem 0163 Ohajiki Game



int main(void)
{
    int i;
    int j;
    int a[25];
    int ohajiki;
    int jiro_turn;
    
    while (1){
        scanf("%d", &i);
        if (i == 0){
            break;
        }

        for (j = 0; j < i; j++){
            scanf("%d", &a[j]);
        }
        
        jiro_turn = 0;
        ohajiki = 32;
        while (1){
            // Y^[
            ohajiki -= ((ohajiki - 1) % 5);
            
            // \
            printf("%d\n", ohajiki);

            // I
            if (ohajiki == 0){
                break;
            }
            
            // Y^[
            ohajiki -= a[jiro_turn];
            jiro_turn = ((jiro_turn + 1) % i);

            // I
            if (ohajiki <= 0){
                printf("0\n");
                break;
            }

            // \
            printf("%d\n", ohajiki);
        }
    }
    
    return (0);
}
