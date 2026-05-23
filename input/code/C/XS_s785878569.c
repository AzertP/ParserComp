int main() {
  int A,B,C ;
  scanf("%d %d %d",&A,&B,&C);
int x;
 if(A==C)
 x=B;
 if(A==B)
 x=C;
 if(B==C)
 x=A;
int ans=x;
 printf("%d\n",ans);
  return 0;
}
