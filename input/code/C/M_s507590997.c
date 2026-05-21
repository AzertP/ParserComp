#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void push(int );
int pop (void);
int top,a[100];
int main()
{
  int a,b;
  char c[100];
  top=0;

  while(scanf("%s",c)!=EOF)
    {
      if(c[0]=='-')
	{
	  b=pop();
	  a=pop();
	  push(a-b);
	}
      else if(c[0]=='+')
	{
	  a=pop();
	  b=pop();
	  push(a+b);
	}
      else if(c[0]=='*')
	{
	  a=pop();
	  b=pop();
	  push(a*b);
	}
      else push(atoi(c));
      
    }
 
  
  printf("%d\n",pop());


  return 0;

  
}
void push(int x )
{
  a[++top]=x;
  
}
int pop (void)
{
  top--;

  return a[top+1];
}
  





