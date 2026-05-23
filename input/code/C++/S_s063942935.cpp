using namespace std;
int v[100];
int main() {
    int n; cin >> n;
    int u, k;
    int a;
    for (int i = 0; i < n; i++) {
        cin >> u >> k;
        for (int j = 0; j < k; j++) {
            cin >> a;
            v[a-1] = 1;
        }
        for (int j = 0; j < n; j++) {
            if (j) cout << " ";
            cout << v[j];
            v[j] = 0;
        }
        cout << endl;
        
    }
    
    return 0;
}

