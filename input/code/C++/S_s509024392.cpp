#include <iostream>

using namespace std;

int main()
{
    int n,x,i=0;
    for(;;){
        cin >> n >> x;
        if(n!=0||x!=0){
            for(int a=0;a<n-2;a++){
                for(int b=a+1;b<n-1;b++){
                    for(int c=b+1;c<n;c++){
                        if(a+b+c+3==x)
                            i++;
                    }
                }
            }
        }
        else break;
        cout << i << endl;
        i=0;
    }

    return 0;
}

