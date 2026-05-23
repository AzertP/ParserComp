
int main(void){
    long int oni_A, hito_B;
    long int v_oni,v_hito;
    long int Time;
    scanf("%ld %ld",&oni_A,&v_oni);
    scanf("%ld %ld",&hito_B,&v_hito);
    scanf("%ld",&Time);
    //10^9 + 10^9+10^9 < long int(8bite) 
    long int Aed,Bed;
    //
    if(oni_A < hito_B){
        Aed = oni_A + Time*v_oni;
        Bed = hito_B + Time*v_hito;
        if(Aed >= Bed){
            printf("YES");
            return 0;
        }else{
            printf("NO");
            return 0;
        }
    }else{
        Aed = oni_A -(Time*v_oni);
        Bed = hito_B - Time*v_hito;
        if(Aed <= Bed){
            printf("YES");
            return 0;
        }else{
            printf("NO");
            return 0;
        }
    }
    return 0;
}
