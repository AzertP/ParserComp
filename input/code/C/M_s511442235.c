#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void insersion_sort(int *A, int N)
{
    int v;
    int j;

    for (int i = 1; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            if (k == N - 1)
            {
                printf("%d\n", A[k]);
            }
            else
            {
                printf("%d ", A[k]);
            }
        }
        v = A[i];
        j = i - 1;
        while (j >= 0 && A[j] > v)
        {
            A[j + 1] = A[j];
            j -= 1;
        }
        A[j + 1] = v;
    }
}

int main(void)
{
    int N;
    int A[100];
    int times;

    scanf("%d", &N);
    for (int i = 0; i < N; i++)
    {
        scanf("%d", &A[i]);
    }

    insersion_sort(A, N);
    for (int i = 0; i < N; i++)
    {
        if (i == N - 1)
        {
            printf("%d\n", A[i]);
        }
        else
        {
            printf("%d ", A[i]);
        }
    }

    return 0;
}
