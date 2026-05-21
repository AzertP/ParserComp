import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main
{

  public static void main(String[] args) throws IOException
  {
    BufferedReader r = new BufferedReader(new InputStreamReader(System.in), 1);
    String s = r.readLine();
    String[] sl = s.split("[\\s]+");
    int n = Integer.parseInt(sl[0]);
    int k = Integer.parseInt(sl[1]);

    int v = 0;

    int a[] = new int[100001];
    s = r.readLine();
    sl = s.split("[\\s]+");

    for(int i = 1; i < n; i++)
    {
      a[i] = Integer.parseInt(sl[i - 1]);
      if(a[i] == 1)
      {
        v = i;
      }
    }

    int c = (v - 1 + k - 2) / (k - 1);

    int z = c * (k - 1) + 1;

    if(z < n) c += (n - z + k - 2) / (k - 1);

    System.out.println(c);
  }

}
