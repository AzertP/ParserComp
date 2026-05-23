using namespace std;
int main(){
    int a,b;
    cin >> a >> b;
    int porda;

    porda = b*2;
    if(porda > a){
        cout << "0" <<endl;
    }else{
    cout << a - porda << endl;
}
}
