#include<bits/stdc++.h>
using namespace std;


int main() 
{
    char S[3];
    for (size_t i = 0; i < 3; i++)
    {
        cin>>S[i];
    }
    int ans=0;
    if(S[0]=='R')
    {
        ++ans;
    }

    for (size_t i = 1; i < 3; i++)
    {
        if(S[i]=='R')
        {
            ++ans;
        }
        else if(ans==0)
        {
            continue;
        }
        else
        {
            break;
        }
        
    }

    cout<<ans<<endl;
    
    return 0;
}