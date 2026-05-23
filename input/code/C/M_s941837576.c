int main(){
  int a,b,c,x,y;
  long ans;
  scanf("%d%d%d%d%d",&a,&b,&c,&x,&y);
  if(a+b<=c*2){
    ans=x*a+y*b;
    printf("%ld\n",ans);
    return 0;
  }
  else{
    if(x>y){
      ans=c*y*2;
      if(a>c*2){
        ans+=c*2*(x-y);
      }
      else{
        ans+=a*(x-y);
      }
    }
    else{
      ans=c*x*2;
      if(b>c*2){
        ans+=c*2*(y-x);
      }
      else{
        ans+=b*(y-x);
      }
    }
    printf("%ld\n",ans);
    return 0;
  }
}
