import java.io.*;
import java.util.*;

class Main {
  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int N = Integer.parseInt(sc.next());
    int[] H = new int[N];
    for (int i = 0; i < N; i++) {
      H[i] = Integer.parseInt(sc.next());
    }

    if (N == 1) {
      System.out.println("Yes");
      return;
    }
    
    if (H[0] > H[1]) {
      H[0]--;
    }
    for (int i = 1; i < N; i++) {
      if (H[i-1] > H[i]) {
        System.out.println("No");
        return;
      }
      if (H[i] > H[i-1]) {
        H[i]--;
      }
    }
    
    if (H[N-1] < H[N-2]) {
      System.out.println("No");
      return;
    }
    
    System.out.println("Yes");
  }
}