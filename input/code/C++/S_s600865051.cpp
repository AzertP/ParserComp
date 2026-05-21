#include <iostream>
#include <string>
using namespace std;
int main(void){
    int r,c;
    cin >> r >> c;
    int Spreadsheet[101][101]={};
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            cin >> Spreadsheet[i][j];
            Spreadsheet[i][c]+=Spreadsheet[i][j];
            Spreadsheet[r][j]+=Spreadsheet[i][j];
        }
    }
    for(int i=0;i<c;i++){
        Spreadsheet[r][c]+=Spreadsheet[r][i];
    }
    for(int i=0;i<r+1;i++){
        for(int j=0;j<c;j++){
            cout << Spreadsheet[i][j] << " ";
        }
        cout << Spreadsheet[i][c] << endl;
    }
}
