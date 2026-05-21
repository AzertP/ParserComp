#include<iostream>
using namespace std;
int map[101][101];
int main()
{
	int h,w;
	cin>>h>>w;
	int n;
	int a[10001]={0};
	cin>>n;
	for(int i=1;i<=n;i++)
	scanf("%d",&a[i]);
	int ans=1;
	int dir=1;
	for(int i=1;i<=h;i++)
	{   
	    if(dir==1)
	    {
    		 for(int j=1;j<=w;j++)
			{   
			    if(a[ans]==0)
			    {
	    			ans++;	
	    		}			
				map[i][j]=ans;
	    			a[ans]--;
                	
			}
    	}
    	else
    	{
	    	for(int j=w;j>=1;j--)
			{   
			    if(a[ans]==0)
			    {
	    			ans++;
	    		}							
			    map[i][j]=ans;
    			a[ans]--;							
			}
	    }
	    dir=-dir;		
	}
	for(int i=1;i<=h;i++)
	{
		for(int j=1;j<w;j++)
		cout<<map[i][j]<<" ";
		cout<<map[i][w]<<endl; 
	} 
} 