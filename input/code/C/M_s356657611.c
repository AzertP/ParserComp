#include <stdio.h>

#define BUILDINGS 4
#define FLOORS 3
#define ROOMS 10

int main(int argc, char *argv[])
{
    const char *const line_separator = "####################\n";
    int counts[BUILDINGS][FLOORS][ROOMS] = {};
    int n;
    scanf("%d\n", &n);
    for (int i = 0; i < n; ++i)
    {
        int building, floor, room, fluctuation;
        scanf("%d %d %d %d\n", &building, &floor, &room, &fluctuation);
        counts[building - 1][floor - 1][room - 1] += fluctuation;
    }
    for (int building = 0; building < BUILDINGS; ++building)
    {
        printf("%s", building ? line_separator : "");
        for (int floor = 0; floor < FLOORS; ++floor)
        {
            for (int room = 0; room < ROOMS; ++room)
            {
                printf("%2d", counts[building][floor][room]);
            }
            printf("\n");
        }
    }
    return 0;
}

