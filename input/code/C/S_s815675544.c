typedef long long ll;
int main(void)
{
  ll a,b,k;
  scanf("%lld %lld %lld",&a,&b,&k);

  if(b-a+1<2*k){
    for(ll i=a;i<=b;i++){
      printf("%lld\n",i);
    }
  }else{
    for(ll i=a;i<a+k;i++){
      printf("%lld\n",i);
    }
    for(ll i=b-k+1;i<=b;i++){
      printf("%lld\n",i);
    }

  }


  return 0;
}
