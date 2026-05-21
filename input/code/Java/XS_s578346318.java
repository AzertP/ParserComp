import java.util.Scanner;

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        char n[] = new char[4];
        n = sc.next().toCharArray();
        sc.close();
        for (int i = 0; i < 3; ++i) {
            if (n[i] == '1') n[i] = '9';
            else if (n[i] == '9') n[i] = '1';
        }
        System.out.print(n);
    }
}