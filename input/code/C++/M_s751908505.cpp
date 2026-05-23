
using namespace std;

int main() {
	int n;
	cin >> n;
	string a;
	cin >> a;
	int counter = 0;
	for (int i = 0; i < n; i++) {
		string b;
		cin >> b;
		bool hantei = false;
		while (b.find(a[0]) != string::npos) {
			int p = b.find(a[0]);
			for (int j = 1; ; j++) {
				if (p+(a.size()-1)*j > b.size()-1) {
					break;
				}
				string c = "";
				c += b[p];
				for (int k = 1; k < a.size(); k++) {
					c += b[p+j*k];
				}
				if (a == c) {
					hantei = true;
					counter++;
					break;
				}
			}
			if (hantei) {
				break;
			}
			b = b.substr(p+1);
		} 
	}
	cout << counter << endl;
	return 0;
}
