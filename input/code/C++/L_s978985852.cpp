
using namespace std;


template <class T>
int __builtin_popcount(T n) { return n ? 1 + __builtin_popcount(n & (n - 1)) : 0; }


const double EPS = 1e-8;
typedef long long ll;
typedef pair<int, int> pint;


int main()
{
	int m, n;
	while (scanf("%d%d", &m, &n), m|n)
	{
		bool alive[1000];
		fill(alive, alive+m, true);
		int p = 0;
		int left = m;
		for (int i = 1; i <= n; ++i)
		{
			char buf[32];
			scanf("%s", buf);
			if (left == 1)
				continue;

			char ans[32];
			if (i % 15 == 0)
				strcpy(ans, "FizzBuzz");
			else if (i % 3 == 0)
				strcpy(ans, "Fizz");
			else if (i % 5 == 0)
				strcpy(ans, "Buzz");
			else
				sprintf(ans, "%d", i);

			if (strcmp(ans, buf))
			{
				alive[p] = false;
				--left;
			}
			do
			{
				p = (p + 1) % m;
			} while (!alive[p]);
		}

		for (int i = 0, j = 0; i < m; ++i)
		{
			if (alive[i])
			{
				printf("%d", i + 1);
				++j;
				if (j == left)
					printf("\n");
				else
					printf(" ");
			}
		}
	}

	return 0;
}
