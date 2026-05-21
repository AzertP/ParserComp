import java.util.*;

public class Main{
    public static void main(String[] args) {
        Scanner sc  = new Scanner(System.in);
        int ac = 0, wa = 0, tle = 0, re = 0;
        int n = Integer.parseInt(sc.nextLine());
        for(int i = 0;i<n;i++){
            String temp = sc.nextLine();
            if(temp.equals("AC")) ac++;
            if(temp.equals("WA")) wa++;
            if(temp.equals("TLE")) tle++;
            if(temp.equals("RE")) re++;
        }
        System.out.println("AC x " + ac);
        System.out.println("WA x " + wa);
        System.out.println("TLE x " + tle);
        System.out.println("RE x " + re);
    }
}