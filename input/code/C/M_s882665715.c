
int main(void)
{
    int a, b, c, min, max;
    scanf("%d", &a);
    scanf("%d", &b);
    scanf("%d", &c);

    if (a < b) {
        if (b < c) {
            min = a;
            max = c;
        }
        else {
            max = b;
            if (a < c) min = a;
            else min = c;
        }
    }
    else {
        if (a < c) {
            min = b;
            max = c;
        }
        else {
            max = a;
            if (b < c) min = b;
            else min = c;
        }
    }

    printf("%d %d\n", min, max);

    return 0;
}
