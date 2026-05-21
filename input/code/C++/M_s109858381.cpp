#include<iostream>
#include<vector>
#define ull unsigned long long int 
using namespace std;


ull hcf(ull a,ull b){
    if(b==0)
    return a;
    return hcf(b,a%b);
}


ull lcm(ull a,ull b){
    
    ull den=hcf(a,b);
    
    ull result=0;
    
    if(a%den==0){
        result=a/den;
        result*=b;
    }
    else{
        result=b/den;
        result*=a;
    }
    return result;
}


int main(){
    ull n;
    cin>>n;
    vector<ull>t(n);
    for(int i=0;i<n;i++)
    {
        cin>>t[i];
    }
    ull result=1;
    for(int i=0;i<n;i++)
    {
        result=lcm(result,t[i]);
    
    }
    cout<<result;

}