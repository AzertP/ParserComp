#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
using namespace std;

int main()
{
    int n,k,cnt=0;
    cin>>n>>k;
    while(n)
    {
        n/=k;
        cnt++;
    }
    cout<<cnt<<endl;
    return 0;
}
