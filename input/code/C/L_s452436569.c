#include <stdio.h>
#include <string.h>

#define STR_MAX 100
#define MAX_NODE 500001

// ノード型定義
typedef struct _NODE{
    int key;
    struct _NODE *right;
    struct _NODE *left;
} NODE;

// ノード用RAM管理
NODE node_pool[MAX_NODE];
int node_cnt;
NODE *make_node(int key);

// ツリー操作
NODE *insert_tree(NODE *root,int k);
void print_tree(NODE *root);
void print_tree_inorder(NODE *node);
void print_tree_preorder(NODE *node);

/////////////////////////////
// メイン処理
int main(){
    int i;
    int m,k;
    NODE *root;
    char buf[STR_MAX];

    node_cnt = 0;
    root = NULL;

    // 命令数を読み込んでループ
    scanf("%d%*c",&m);
    for( i=0; i<m; i++){
        // 命令を読み込んで処理
        fgets(buf,STR_MAX,stdin);
        if(strncmp(buf,"insert",6)==0){
            // insert
            sscanf(buf,"%*s %d",&k);
            root = insert_tree(root,k);
        }else if(strncmp(buf,"print",5)==0){
            // print
            print_tree(root);
        }else{
            //処理なし
        }
    }

    return 0;
}


/////////////////////////////
// ツリー操作

// 要素追加
// ツリーのルートを返す
NODE *insert_tree(NODE *root,int k){
    NODE *new_root = root;
    NODE *y = NULL; // x の親
    NODE *x = root;
    NODE *z = make_node(k);

    while( x != NULL ){
        y = x; // 親を設定
        if( z->key < x->key ){
            x = x->left; // 左の子へ移動
        }else{
            x = x->right; // 右の子へ移動
        }
    }

    if( y == NULL){ // T が空の場合
        new_root = z;
    }else if( z->key < y->key){
        y->left = z; // z を y の左の子にする
    }else{
        y->right = z; // z を y の右の子にする
    }

    return new_root;
}

// 内容表示
void print_tree(NODE *root){
    print_tree_inorder(root);
    print_tree_preorder(root);
}
// 中間順巡回
void print_tree_inorder(NODE *node){
    static NODE *stack_node[MAX_NODE*2];
    static int   stack_type[MAX_NODE*2];// 0:巡回 1:表示
    int sp;
    NODE *tmp_node;

    stack_node[0]=node;
    stack_type[0]=0;
    sp = 1;
    while(sp>0){
        sp--;
        tmp_node = stack_node[sp];
        if(tmp_node==NULL)continue;
        switch(stack_type[sp]){
            case 0: //巡回
                stack_node[sp]=tmp_node->right;
                stack_type[sp]=0; //巡回
                sp++;
                stack_node[sp]=tmp_node;
                stack_type[sp]=1; //表示
                sp++;
                stack_node[sp]=tmp_node->left;
                stack_type[sp]=0; //巡回
                sp++;
                break;
            case 1: //表示
                printf(" %d",tmp_node->key);
                break;
            default:
                break;
        }
    }
    printf("\n");
}

// 先行順巡回
void print_tree_preorder(NODE *node){
    static NODE *stack[MAX_NODE];
    int sp;
    NODE *tmp_node;

    stack[0]=node;
    sp = 1;
    while(sp>0){
        tmp_node = stack[--sp];
        if(tmp_node==NULL)continue;
        stack[sp++]=tmp_node->right;
        stack[sp++]=tmp_node->left;
        printf(" %d",tmp_node->key);
    }
    printf("\n");
}

//////////////////////
//ノード用RAM管理

//新規ノード作成
NODE *make_node(int key){
    NODE *node = &node_pool[node_cnt];
    node_cnt++;
    node->key   = key;
    node->right = NULL;
    node->left  = NULL;
    return node;
}


