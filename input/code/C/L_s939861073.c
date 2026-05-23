int main()
{
    long long int n,k; //nk
    long long int r,s,p; //
    long long int ans=0;
    char t[200000],ne[200000];
    scanf("%lld %lld",&n,&k);
    scanf("%lld %lld %lld",&r,&s,&p);
    scanf("%s",t);
    strcpy(ne,t);
    //printf("%s\n",ne);
    for(long long int i =0;i<n;i++)
    {
        if(ne[i]=='r')
            ne[i]='p';
        else if(ne[i]=='s')
            ne[i]='r';
        else
            ne[i]='s';
    }
    //printf("%s\n",ne);
    for(long long int i =k;i<n;i++)
    {
        if(ne[i]==ne[i-k])
        {
            if((i+k)<n)
        {
            if(ne[i]=='r')
            {
                if(ne[i+k]=='s')
                    ne[i]='p';
                else if(ne[i+k]=='p')
                    ne[i]='s';
                else
                    ne[i]='s';
            }
            else if(ne[i]=='s')
            {
                if(ne[i+k]=='r')
                    ne[i]='p';
                else if(ne[i+k]=='p')
                    ne[i]='r';
                else
                    ne[i]='r';
            }
            else if(ne[i]=='p')
            {
                if(ne[i+k]=='r')
                    ne[i]='s';
                else if(ne[i+k]=='s')
                    ne[i]='r';
                else
                    ne[i]='r';
            }
        }
            else
            {
                if(ne[i]=='r')
                {
                        ne[i]='s';
                }
                else if(ne[i]=='s')
                {
                        ne[i]='r';
                }
                else if(ne[i]=='p')
                {
                        ne[i]='r';
                }
            }
            
        }
    }
   //printf("%s\n",ne);
    for(long long int i =0;i<n;i++)
    {
        if(ne[i]=='r'&&t[i]=='s')
            ans+=r;
        else if(ne[i]=='s'&&t[i]=='p')
            ans+=s;
        else if(ne[i]=='p'&&t[i]=='r')
            ans+=p;
    }
    printf("%lld\n",ans);
    return 0;
}
