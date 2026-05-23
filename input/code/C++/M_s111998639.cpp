using namespace std;
int a[4][4],v[4][4];
int b[105];
int n;
int main()
{
	for(int i=1;i<=3;i++)
	{
		for(int j=1;j<=3;j++)
		{
			cin>>a[i][j];
		}
	}
	cin>>n;
	for(int i=0;i<n;i++)
	{
		cin>>b[i];
	}
	memset(v,0,sizeof(v));
	for(int i=0;i<n;i++)
	{
		int flag=0;
		for(int j=1;j<=3;j++)
		{
			
			for(int k=1;k<=3;k++)
			{
				if(a[j][k]==b[i])
				{
					v[j][k]=1;
					flag=1;
					break;
				}	
			}
			if(flag==1)	break;
		}
	}
	int flag=0;
	for(int i=1;i<=3;i++)
	{
		if(v[i][1]==1&&v[i][2]==1&&v[i][3]==1||v[1][i]==1&&v[2][i]==1&&v[3][i]==1)
		{
			flag=1;
		}
	}
	if(v[1][1]==1&&v[2][2]==1&&v[3][3]==1||v[1][3]==1&&v[2][2]==1&&v[3][1]==1)
		flag=1;
	if(flag)
		cout<<"Yes"<<endl;
	else
		cout<<"No"<<endl;
}
