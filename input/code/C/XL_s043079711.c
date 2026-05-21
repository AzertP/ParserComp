#include <stdio.h>

typedef struct{
    int p, k, l, r;
}node;

node nodes[500001];

int ino_f = 0;
int next_id = 0;

int next(int z);


void ino(int id)
{
    if(id == -1)
        return ;
    ino(nodes[id].l);
    printf(" %d", nodes[id].k);
    ino(nodes[id].r);
}

void find(int z)
{
    int x = 0;
    while(x >= 0 && nodes[x].k != z){
        if(z < nodes[x].k){
            x = nodes[x].l;
        }else{
            x = nodes[x].r;
        }
    }

    if(nodes[x].k == z)
        printf("yes\n");
    else
        printf("no\n");
}

void insert(int index, int z)
{
    int x = 0, y;

    nodes[index].k = z;
    nodes[index].l = -1;
    nodes[index].r = -1;

    if(index == 0)
        return ;

    while(x >= 0){
        y = x;
        if(z < nodes[x].k){
            x = nodes[x].l;
        }else{
            x = nodes[x].r;
        }
    }

    if(z < nodes[y].k){
        nodes[y].l = index;
    }else{
        nodes[y].r = index;
    }
    nodes[index].p = y;
}

int find_d(int z)
{
    int x = 0;
    while(x >= 0 && nodes[x].k != z){
        if(z < nodes[x].k){
            x = nodes[x].l;
        }else{
            x = nodes[x].r;
        }
    }

    if(nodes[x].k == z)
        return x;
    else
        return -1;
}

void del_sub(int n)
{
    if(nodes[n].l == -1 && nodes[n].r == -1){
        if(nodes[nodes[n].p].l == n){
            nodes[nodes[n].p].l = -1;
        }else{
            nodes[nodes[n].p].r = -1;
        }
    }else if(nodes[n].l == -1 && nodes[n].r != -1){//only on right
        if(nodes[nodes[n].p].l == n){
            nodes[nodes[n].p].l = nodes[n].r;
        }else{
            nodes[nodes[n].p].r = nodes[n].r;
        }
        nodes[nodes[n].r].p = nodes[n].p;
    }else if(nodes[n].l != -1 && nodes[n].r == -1){
        if(nodes[nodes[n].p].l == n){
            nodes[nodes[n].p].l = nodes[n].l;
        }else{
            nodes[nodes[n].p].r = nodes[n].l;
        }
        nodes[nodes[n].l].p = nodes[n].p;
    }
}

void delete(int z)
{
    int n = find_d(z);

    if(nodes[n].l == -1 && nodes[n].r == -1){
        //printf("%d: 1\n", z);
        if(nodes[nodes[n].p].l == n){
            nodes[nodes[n].p].l = -1;
        }else{
            nodes[nodes[n].p].r = -1;
        }
    }else if(nodes[n].l == -1 && nodes[n].r != -1){//only on right
        //printf("%d: 2\n", z);
        if(nodes[nodes[n].p].l == n){
            nodes[nodes[n].p].l = nodes[n].r;
        }else{
            nodes[nodes[n].p].r = nodes[n].r;
        }
        nodes[nodes[n].r].p = nodes[n].p;
    }else if(nodes[n].l != -1 && nodes[n].r == -1){
        //printf("%d: 3\n", z);
        if(nodes[nodes[n].p].l == n){
            nodes[nodes[n].p].l = nodes[n].l;
        }else{
            nodes[nodes[n].p].r = nodes[n].l;
        }
        nodes[nodes[n].l].p = nodes[n].p;
    }else if(nodes[n].l != -1 && nodes[n].r != -1){
        int y;
        ino_f = 0;
        y = next(z);
        //printf("%d: 4\n", z);
        //printf("next : %d\n", y);
        nodes[n].k = nodes[y].k;
        del_sub(y);
    }

}

void pre(int id)
{
    if(id == -1)
        return ;
    printf(" %d", nodes[id].k);
    pre(nodes[id].l);
    pre(nodes[id].r);
}

void ino_n(int id, int z)
{
    if(id == -1)
        return ;

    ino_n(nodes[id].l, z);
    if(nodes[id].k == z && !ino_f){
        ino_f = 1;
    }else if(ino_f){
        next_id = id;
        ino_f = 0;
    }
    ino_n(nodes[id].r, z);
}

int next(int z){
    ino_n(0, z);
    return next_id;
}

int main(void)
{
    int i, n;
    int index = 0, z;
    char buf[16];
    scanf("%d", &n);

    for(i = 0; i < n; i++){
        scanf("%s", buf);
        if(buf[0] == 'i'){
            scanf("%d", &z);
            insert(index++, z);
        }else if(buf[0] == 'p'){
            if(index > 0){
                ino(0);
                printf("\n");
                pre(0);
                printf("\n");
            }
        }else if(buf[0] == 'f'){
            scanf("%d", &z);
            find(z);
        }else if(buf[0] == 'd'){
            scanf("%d", &z);
            delete(z);
        }
    }

    return 0;
}