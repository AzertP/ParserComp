import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 *
 * @author CompuWare
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        Scanner in = new Scanner(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        BTripleDots solver = new BTripleDots();
        solver.solve(1, in, out);
        out.close();
    }

    static class BTripleDots {
        public void solve(int testNumber, Scanner in, PrintWriter out) {
            int K = in.nextInt();
            String S = in.next();

            if (S.length() > K) {
                out.print(S.substring(0, K) + "...");
            } else {
                out.print(S);
            }
        }

    }
}

