// Ver19.03
int upll(const void *a, const void *b) { return *(ll *)a < *(ll *)b ? -1 : *(ll *)a > *(ll *)b ? 1 : 0; }
int downll(const void *a, const void *b) { return *(ll *)a < *(ll *)b ? 1 : *(ll *)a > *(ll *)b ? -1 : 0; }
void sortup(ll *a, int n) { qsort(a, n, sizeof(ll), upll); }
void sortdown(ll *a, int n) { qsort(a, n, sizeof(ll), downll); }

int a[100], b[100], c[100];
int main()
{
  int n, ans = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++)
    scanf("%d", a + i);
  for (int i = 0; i < n; i++)
    scanf("%d", b + i);
  for (int i = 0; i < n - 1; i++)
    scanf("%d", c + i);
  for (int i = 0; i < n; i++)
  {
    if (i >= 1)
    {
      if (a[i] - a[i - 1] == 1)
        ans += c[a[i - 1] - 1];
    }
    ans += b[a[i] - 1];
  }
  printf("%d\n", ans);
}
