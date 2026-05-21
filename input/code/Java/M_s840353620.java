import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {

  public static void main(String[] args) throws Exception {
    try (final var stdin = new InputStreamReader(System.in);
         final var br = new BufferedReader(stdin)) {
      final var n = Integer.parseInt(br.readLine());

      switch (n % 10) {
        case 2:
        case 4:
        case 5:
        case 7:
        case 9:
          System.out.println("hon");
          break;
        case 0:
        case 1:
        case 6:
        case 8:
          System.out.println("pon");
          break;
        case 3:
          System.out.println("bon");
          break;
      }
    }
  }
}
