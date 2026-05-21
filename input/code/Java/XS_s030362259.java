import java.util.*;

class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        String keyword = scan.next();
        int count = 0;

        while (true) {
            String text = scan.next();
            if (text.equals("END_OF_TEXT")) break;
            if (text.equalsIgnoreCase(keyword)) count++;
        }
        System.out.println(count);
    }
}