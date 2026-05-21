#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define COMMAND_LENGTH 32

typedef struct _DoublyList {
    struct _DoublyList* head;
    struct _DoublyList* tail;
    int key;
} DoublyList;

char command_buffer[COMMAND_LENGTH];

void get_command(int* cmd_id, int* key);
//DoublyList* new_list(void);
void append_list_head(DoublyList** haed, DoublyList** tail, int key);
void delete_list(DoublyList** head, DoublyList** tail, int key);
void delete_list_head(DoublyList** head, DoublyList** tail);
void delete_list_tail(DoublyList** head, DoublyList** tail);
void destruct_list(DoublyList** head, DoublyList** tail);
void print_list(DoublyList* head);

void main(void)
{
    int nn;
    fgets(command_buffer, COMMAND_LENGTH, stdin);
    nn = atoi(command_buffer);
    assert(nn != 0);
    //printf("nn = %d\n", nn);

    DoublyList* head = NULL;
    DoublyList* tail = NULL;
    
    int n;
    for (n = 0; n < nn; n+=1) {
        int cmd_id;
        int key;
        get_command(&cmd_id, &key);
        switch (cmd_id) {
          case 0:
            append_list_head(&head, &tail, key);
            break;
          case 1:
            delete_list(&head, &tail, key);
            break;
          case 2:
            delete_list_head(&head, &tail);
            break;
          case 3:
            delete_list_tail(&head, &tail);
            break;
          default:
            assert(0);
            break;
        }
        if (head != NULL) {
            assert(head->head == NULL);
            assert(tail->tail == NULL);
        }
        //print_list(head);
    }
    destruct_list(&head, &tail);
}

void get_command(int* cmd_id, int* key)
{
    fgets(command_buffer, COMMAND_LENGTH, stdin);
    if (strncmp(command_buffer, "insert ", 7) == 0) {
        *cmd_id = 0;
    } else if (strncmp(command_buffer, "delete ", 7) == 0) {
        *cmd_id = 1;
    } else if (strncmp(command_buffer, "deleteF", 7) == 0) {
        *cmd_id = 2;
    } else if (strncmp(command_buffer, "deleteL", 7) == 0) {
        *cmd_id = 3;
    } else {
        assert(0);
        *cmd_id = 99;
    }
    if ((*cmd_id) == 0 || (*cmd_id == 1)) {
        strtok(command_buffer, " ");
        char* key_str = strtok(NULL, " ");
        assert(key_str != NULL);
        *key = atoi(key_str);
    } else {
        *key = -1;
    }
    //printf("cmd_id = %d\n", *cmd_id);
    //printf("key    = %d\n", *key);
}

DoublyList* new_list(void)
{
    DoublyList* list = (DoublyList*)malloc(sizeof(DoublyList));
    //fprintf(stderr, "a: 0x%016lx\n", (unsigned long)list);
    assert(list != NULL);
    //list->tail = NULL;
    list->head = NULL;
    return list;
}

void free_list(DoublyList* list)
{
    //fprintf(stderr, "d: 0x%016lx\n", (unsigned long)list);
    free(list);
}

void append_list_head(DoublyList** head, DoublyList** tail, int key)
{
    DoublyList* new_head = new_list();
    new_head->key  = key;
    if (*head == NULL) { 
        assert(*tail == NULL); 
        *tail = new_head;
        new_head->tail = NULL;
    } else {
        assert(*tail != NULL); 
        (*head)->head = new_head;
        new_head->tail = *head;
    }
    *head = new_head;
}

void delete_list(DoublyList** head, DoublyList** tail, int key)
{
    if (*head == NULL) {
        assert(*tail == NULL);
        return;
    }
    assert(*tail != NULL);
    if ((*head)->key == key) {
        delete_list_head(head, tail);
        return;
    }
    DoublyList* lp;
    int is_hit = 0;
    for(lp = *head; lp->tail != NULL; lp = lp->tail) {
        if (lp->tail->key != key) {
            continue;
        }
        if (lp->tail->tail == NULL) {
            break;
        }
        DoublyList* list = lp->tail;
        lp->tail = lp->tail->tail;
        lp->tail->head = lp;
        free_list(list);
        is_hit = 1;
        break;
    }
    //printf("delete list\n");
    if ((is_hit == 0) && ((*tail)->key == key)) {
        delete_list_tail(head, tail);
        return;
    }
}

void delete_list_head(DoublyList** head, DoublyList** tail)
{
    //printf("delete head\n");
    if (*head == NULL) {
        return;
    }
    assert((*head)->head == NULL);
    DoublyList* list = *head;
    if ((*head)->tail != NULL) {
        (*head)->tail->head = NULL;
        (*head) = (*head)->tail;
    } else {
        *head = NULL;
        *tail = NULL;
    }
    free_list(list);
}

void delete_list_tail(DoublyList** head, DoublyList** tail)
{
    //printf("delete tail\n");
    if (*tail == NULL) {
        return;
    }
    assert((*tail)->tail == NULL);
    DoublyList* list = *tail;
    if ((*tail)->head != NULL) {
        (*tail)->head->tail = NULL;
        (*tail) = (*tail)->head;
    } else {
        *head = NULL;
        *tail = NULL;
    }
    free_list(list);
}

void destruct_list(DoublyList** head, DoublyList** tail)
{
    while (*head != NULL) {
        if ((*head)->tail != NULL) {
            printf("%d ", (*head)->key);
        } else {
            printf("%d\n", (*head)->key);
        }
        //fprintf(stderr, "> 0x%016lx\n", (unsigned long)(*head)->tail);
        delete_list_head(head, tail);
    }
    //printf("\n");
}

void print_list(DoublyList* head)
{
    DoublyList* lp;
    for (lp = head; lp->tail != NULL; lp = lp->tail) {
        printf("%d ", lp->key);
    }
    if (lp != NULL) {
        printf("%d\n", lp->key);
    }
}


