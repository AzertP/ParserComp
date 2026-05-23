using namespace std;
int main(){
    int W,H,x,y,r;
    cin >> W >> H >> x >> y >> r;
    if(r <= x && r <= y && x+r <= W && y+r <= H){
        cout << "Yes" <<endl;
    }else{
        cout << "No" <<endl;
    }
    
}
