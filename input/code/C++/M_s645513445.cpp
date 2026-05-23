
using namespace std;

int main()
{
    int n,i,j,u,k;
    cin >> n;
    int Adj[n][n];
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            Adj[i][j]=0;
    for(i=0;i<n;i++)
    {
        cin >> u >> k;
        int v[k];
        for(j=0;j<k;j++)
        {
            cin >> v[j];
            Adj[u-1][v[j]-1]=1;
        }
    }
    for(i=0;i<n;i++)
    {
        for(j=0;j<n-1;j++)
        {
            cout << Adj[i][j] << ' ';
        }
        cout << Adj[i][n-1] << endl;
    }
    return 0;
}

