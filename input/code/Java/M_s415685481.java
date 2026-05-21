import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        long N = scanner.nextLong();
        long A = scanner.nextLong();
        long B = scanner.nextLong();

        long answer = 0;
        if (A >= N) {
            answer = N;
        } else if (A + B > N) {
            answer = A;
        } else {
            if (A + B != 0) {
                long numberOfBToAdd = N / (A + B);
                answer = numberOfBToAdd * A;
                long additional = N % (A + B);
                if (A != 0) {
                    if (additional >= A) {
                        answer += A;
                    } else {
                        answer += additional;
                    }
                }
            }
        }

        System.out.println(answer);
    }
}