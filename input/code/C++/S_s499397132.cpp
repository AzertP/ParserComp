#include <bits/stdc++.h>
using namespace std;
int main()
{
    int n;
    cin >> n;
    vector<long long int> v;
    for(int i = 0;i < n;i++)
    {
        int x;
        cin >> x;
        v.push_back(x);
    }
    while(v.size() > 1)
    {
        sort(v.begin(),v.end());
        for(int i = 1;i < v.size();i++)
        {
            v.at(i) %= v.at(0);
            if(v.at(i) == 0)
            {
                swap(v.at(i),v.at(v.size() - 1));
                v.pop_back();
                i -= 1;
            }
        }
    }
    cout << v.at(0);

    return 0;
}