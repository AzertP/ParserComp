/* ex 9_1
   cy_an  */

//

//
typedef struct stack{
    int size;
    int count;
    char data[SIZE];
    char *top;
} stack;

//
void initialize_stack(stack *stk){
    stk->count = 0;
    stk->size = SIZE;
    stk->top = &(stk->data[SIZE]);
}

//
void push(stack *stk, char elem){
    if(stk->count >= stk->size){
        fprintf(stderr, "Error: stack overflow. (x=%c)\n", elem);
        exit(1);
    }
    else{
        stk->top--;
        *(stk->top) = elem;
        stk->count++;
    }
}

//
char pop(stack *stk){
    char latest;
    if(stk->count <= 0){
        fprintf(stdout, "Error: stack underflow.\n");
        exit(1);
    }
    else{
        latest = *(stk->top);
        stk->top++;
        stk->count--;
        return latest;
    }
}

int main(void){
    stack input, output;
    char arr[SIZE];
    int i=0;
    
    //
    initialize_stack(&input);
    initialize_stack(&output);

    //
    scanf("%s", arr);
    while(arr[i] != '\0'){
        if(arr[i] == 'B'){
            if(input.count > 0){
                pop(&input);
            }
        }else{
            push(&input, arr[i]);
        }
        i++;
    }
    // 
    while(input.count > 0){
        push(&output, pop(&input));
    }
    // 
    while(output.count > 0){
        printf("%c", pop(&output));
    }
    return 0;
}
