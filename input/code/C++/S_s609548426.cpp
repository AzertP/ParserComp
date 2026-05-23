using namespace std;

int main(){
    string s1,s2;
    cin >> s1 >> s2;

    s2.pop_back();

    if(s1 == s2)cout << "Yes" << endl;
    else cout << "No" << endl;
     
    return 0;
}
