

using namespace std;

int main() {
    int a,b;
    cin >> a >> b;
    int o = max(a * 2 - 1,max(a + b,b * 2 - 1));
    cout << o;
}
