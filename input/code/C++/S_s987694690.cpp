using namespace std;

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> h(n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &h.at(i));
    }

    sort(h.begin(), h.end());

    auto itr = lower_bound(h.begin(), h.end(), k);
    int ans = distance(itr, h.end());

    cout << ans << endl;

    return 0;
}
