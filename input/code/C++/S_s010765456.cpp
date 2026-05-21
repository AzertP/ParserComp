#include <iostream>
#include <vector>
using namespace std;



int main() {
	int N; cin >> N;
	vector<int> A(N);
	for (int i = 0; i < N; i++) cin >> A[i];
	uint64_t count = 0;
	int len = 0;
	int tmp = 0;
	for (int i = 0; i < N; i++) {
		while (len > 0 && (tmp & A[i])) {
			len--;
			tmp -= A[i - len - 1];
		}
		if ((tmp & A[i]) == 0) {
			len++;
			count += len;
			tmp += A[i];
		}
	}
	cout << count << endl;

	
}