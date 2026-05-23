using namespace std;
//#pragma gcc optimize("O3")
//#pragma gcc optimize("unroll-loops")

LL trailingZeroes(LL n) {
    LL pows = log(n)/log(5),ans=0;
    for(LL i=1; i<=pows; i++) {
        LL temp = 2*round(pow(5,i));
        if(temp>n) break;
        ans += n/temp;
    }
    return ans;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr); cout.tie(nullptr);
        //freopen(ipf,"r",stdin);
        //openFile(ipf);
        //closeConsole;
        //freopen(opf,"w",stdout);

    LL n;
    cin >> n;
    if(n&1) cout << 0 << "\n";
    else cout << trailingZeroes(n) << "\n";

        //openFile(opf);
        //Sleep(3000);
        //closeP("notepad.exe");
        //closeConsole;
}
