import java.io.BufferedInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
 
public class Main{
	public static void main(String[] args) {
        Scanner scan = new Scanner(new BufferedInputStream(System.in));
        List<String> output = new ArrayList<String>();
        while (scan.hasNext()) {
            String input = scan.nextLine();
            String[] inarr = input.split(" ");
            int x = Integer.parseInt(inarr[0]);
            int y = Integer.parseInt(inarr[1]);
            if (x >= 0 && x <= 10000 && y >= 0 && y <= 10000) {
                if (x == 0 && y == 0) {
                    break;
                } else {
                    if (x < y) {
                        String inVal = String.valueOf(x) + " " + String.valueOf(y);
                        output.add(inVal);
                    } else if (x > y) {
                        String inVal = String.valueOf(y) + " " + String.valueOf(x);
                        output.add(inVal);
                    } else {
                        output.add(input);
                    }
                }
            }
        }
        if (output.size() <= 3000) {
            for (String out : output) {
                System.out.println(out);
            }
        }
    }
 
}