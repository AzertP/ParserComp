#include <iostream>
using namespace std;

int main(){
    string s;
    cin >>s;
  //cout <<s.size() <<endl;
 
    if(s[s.size()-1]=='2' or s[s.size()-1]=='4' or s[s.size()-1]=='5' 
       or s[s.size()-1]=='7' or s[s.size()-1]=='9')cout <<"hon" <<endl;
else if(s[s.size()-1]=='0' or s[s.size()-1]=='1' or s[s.size()-1]=='6' or s[s.size()-1]=='8' )cout <<"pon" <<endl;
else cout <<"bon" <<endl;
//*/
return 0;
}
