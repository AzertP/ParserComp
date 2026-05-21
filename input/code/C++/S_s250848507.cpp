#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;

int main(){
    int k,x;
    cin>>k>>x;
    for(int i=x-k+1;i<=x+k-1;i++){
        if(i!=x-k+1) cout<<" ";
        cout<<i;
    }
    cout<<endl;
    return 0;
}