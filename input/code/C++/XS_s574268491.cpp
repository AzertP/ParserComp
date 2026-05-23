char a[1000];
using namespace std;
int main()
{
	gets(a);
	cout<<a[0]<<strlen(a)-2<<a[strlen(a)-1];
	return 0;
}
