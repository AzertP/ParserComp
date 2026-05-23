

char dec[MAX_ORDER][MAX];

int getChar_ID(char c){
  if(c == 'A'){
    return 1;
  }else if(c == 'C'){
    return 2;
  }else if(c == 'G'){
    return 3;
  }else if(c == 'T'){
    return 4;
  }else return -1;
}

long analysis(char *str){
  int len = strlen(str);
  int i;
  int id=0;
  int k=1;
  //printf("in to analysis\n" );
  for(i=0; i<len; i++){
    id += k * getChar_ID(str[len-i-1]);
    k *= 4;
    //printf("len-i=%d\n",len-i );
    //printf("Now ID: %d\n", id);
  }
  return id;
}

void insert(int id, char *str){
  strcpy(dec[id],str);
}

void find(int id,char *str){
  if(0==strcmp(dec[id],str)){
    printf("yes\n" );
  }else printf("no\n" );
}

int main() {
  int n;
  char str[MAX];
  char com[7];
  int i;

  scanf("%d",&n);
  for(i=0; i<n; i++){
    scanf("%s %s",com,str);
    if(com[0]=='i'){
      insert(analysis(str),str);
    }else if(com[0]=='f'){
      find(analysis(str),str);
    }else printf("Command is illegal\n");
  }
  return 0;
}

