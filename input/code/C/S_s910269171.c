int main()
{
  int i, j, m, n, cnt1, cnt2=0, ans;

  scanf("%d%d", &m , &n);

  for(i = 0; i < m; i++){
    cnt1 = 0;
    
    for(j = 0; j < n; j++){
      scanf("%d", &ans);
      if(ans == 1) cnt1++;
      
    }
    if(cnt2 < cnt1) cnt2 = cnt1;
    
  }

  printf("%d\n", cnt2);

  return 0; 

}
