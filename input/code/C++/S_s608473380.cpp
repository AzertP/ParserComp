using namespace std;

int main(){
    long long X;
    long long sum=100;
    cin>> X ;

    for(int i=1; i<X; i++){
        sum+=(sum/100);

        if(sum>=X){
            cout << i <<endl;
            break;
        }
    }
    
    
}
