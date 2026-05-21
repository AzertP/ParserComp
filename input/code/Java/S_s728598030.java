import java.util.Scanner;

class Main {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        String p = sc.nextLine();
        p = p + p;
        String s = sc.nextLine();

        int lenp = p.length();
        int lens = s.length();

        boolean result = false;
        for (int i = 0; i <= lenp - lens; i++) {
            if (p.substring(i, i + lens).equals(s)) {
                result = true;
            }
        }
        System.out.println((result ? "Yes" : "No"));

    }



}
