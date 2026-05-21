// B.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include<cmath>
#include<string>

using namespace std;

int main()
{
    string s;

    cin >> s;
    int min = 10000;
    for (int i = 0; i < s.size() - 2; i++) {
        string s2 = s.substr(i, 3);
        int x = atoi(s2.c_str());
        int y = abs(753 - x);

        if (min > y)min = y;

        
    }
    cout << min << endl;

    return 0;
}

