#include <iostream>
#include <string.h> 

using namespace std;



int main(void){

  int x;
  int count = 0;

  do{
   cin >> x;
   count++;
   if(x != 0)
      cout << "Case " << count << ": " << x << endl;
  }while(x != 0);

return 0;
}