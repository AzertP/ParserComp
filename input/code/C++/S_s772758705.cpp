#include<iostream>
using namespace std;
int main()
{
    int a,b;
  while(cin>>a>>b)
  {
    if(a%b==0)
        cout<<"0"<<endl;
    else
    {
        int c=a%b;
        if(c==1)
            cout<<c<<endl;
        else
        {
            while(c>1)
            {
                c--;
            }
            cout<<c<<endl;
        }
    }
  }
    return 0;
}
