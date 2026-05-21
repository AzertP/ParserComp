import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
public class Main {
  public static void main(String[] args) {
    Main m = new Main();
    m.run();
  }
  private void run() {
    String s = null;
    try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in));) {
      s = br.readLine();
    } catch (IOException e) {
      e.printStackTrace();
    }
    s = s.replaceAll("[ABDEG-Z]","");
    if (s.contains("CF")) {
      System.out.println("Yes");
    } else {
      System.out.println("No");
    }
  }
}