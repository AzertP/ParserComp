#include <bits/stdc++.h>
using namespace std;

int main(){
	int N;
	long long K;
	cin >> N >> K;
	vector<long long> A(N);
	for(auto &i:A)cin >> i;
	sort(A.begin(),A.end());
	
	auto low=lower_bound(A.begin(),A.end(),0);
	auto up=upper_bound(A.begin(),A.end(),0);
	
	int nega=(int)(low-A.begin());
	int zero=(int)(up-low);
	int posi=N-(nega+zero);
	
	if(K<=(long long)nega*posi){
		//midより小さいものがK個未満であるかどうか
		long long ok=-1000000000000000000LL,ng=0;
		while(abs(ok-ng)>1){
			long long mid=(ok+ng)/2;
			long long c=0;
			for(int i=0;i<nega;i++){
				c+=(long long)(A.end()-upper_bound(A.begin(),A.end(),(-mid)/(-A[i])));
			}
			
			if(c<K)ok=mid;
			else ng=mid;
		}
		
		cout << ok << endl;
		
	}else if(K<=(long long)nega*posi+(long long)zero*(N-zero)+(long long)zero*(zero-1)/2){
		cout << "0" << endl;
	}else{
		K-=(long long)nega*posi+(long long)zero*(N-zero)+(long long)zero*(zero-1)/2;
		//midより小さいものがK個未満であるかどうか
		long long ok=1,ng=1234567890123456789LL;
		while(abs(ok-ng)>1){
			long long mid=(ok+ng)/2;
			long long c=0;
			for(int i=0;i<nega;i++){
				c+=(long long)(A.begin()-lower_bound(A.begin()+i+1,A.begin()+nega,((mid-1)/(-A[i]))*(-1)))+nega;
			}
			for(int i=N-posi;i<N;i++){
				c+=(long long)(upper_bound(A.begin()+nega+zero,A.begin()+i,(mid-1)/A[i])-A.begin())-(nega+zero);
			}
			if(c<K)ok=mid;
			else ng=mid;
		}
		
		cout << ok << endl;
		
	}
	
	return 0;
}