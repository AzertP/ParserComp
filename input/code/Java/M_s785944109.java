import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

class Main {

    public static void main(String[] args) throws Exception {

        try {

            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

            String[] times = br.readLine().split(" ");
            int now = Integer.parseInt(times[0]);
            int contestStartFromNow = Integer.parseInt(times[1]);

            int contestStratTime = now + contestStartFromNow;

            // convert to 24-hour display
            if (contestStratTime < 24) {
                System.out.println(contestStratTime);
            } else {
                System.out.println(contestStratTime - 24);
            }

        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            System.exit(0);
        } catch (final Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }
}