
int main() {
  int n;
  int i;
  int data[100];
  int in;
  int temp;

  int s, g;
  while(1) {
    scanf("%d", &n);
    if(n == 0) break;
    for(i = 0; i < n; i++) scanf("%d", &data[i]);
    scanf("%d", &in);

    s = 0;
    g = n-1;
    for(i = 1; ; i++) {
      temp = (s + g) / 2;
      if(data[temp] == in) break;
      if(data[temp] > in) {
	g = temp-1;
      } else if(data[temp] < in) {
	s = temp+1;
      }
      if(g-s+1 <= 0) break;
    }
    printf("%d\n", i);
  }
  return 0;
}
