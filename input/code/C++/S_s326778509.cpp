using namespace std;

int main() {
    int n, d, x; cin >> n >> d >> x;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    
    int sum = 0;
    for (int i = 0; i < n; i++) {
        //sum += d + a[i] * (d-1);
        sum += (d-1) / a[i] + 1;
    }
    cout << sum + x << endl;

    return 0;
    
}
