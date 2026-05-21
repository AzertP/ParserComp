import java.util.*;
import java.io.*;
import java.lang.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String s = sc.next();

        int len = s.length();
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i<len; i++){
            if (i%2 == 0){
                ans.append(s.charAt(i));
            }
        }
        System.out.println(ans);
    }
}