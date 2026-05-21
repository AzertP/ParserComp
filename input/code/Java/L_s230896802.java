/*
    Written by Kabir Kanha Arora
    @kabirkanha
 */

import java.util.*;

public class Main {
    public static void main(String[] args) {

        // Input
        Scanner scanner = new Scanner(System.in);
        long n = scanner.nextLong();
        long x = scanner.nextLong();
        long m = scanner.nextLong();

        HashMap<Long, Long> map = new HashMap<>();
        map.put(x, 0L);
        long curr = x;
        boolean zeroFlag = false;
        long loopStart = 0;
        long loopSize = 1;
        long loopStartElement = x;
        long sum = x;

        for (long i = 1; i < 1e7 && i < n; ++i) {
            curr = ((curr % m) * (curr % m)) % m;
            if (curr == 0) {
                zeroFlag = true;
                System.out.println(sum);
                break;
            }
            if (map.containsKey(curr)) {
                loopStart = map.get(curr);
                loopStartElement = curr;
                loopSize = i - loopStart;
                break;
            } else {
                sum += curr;
                map.put(curr, i);
                loopSize++;
            }
        }
        if (!zeroFlag) {

            // Elements before start of loop
            sum = 0;
            curr = x;
            long i;
            if (loopStart > 0)
                sum += x;
            for (i = 1; i < loopStart; ++i) {
                curr = ((curr % m) * (curr % m)) % m;
                sum += curr;
            }

            // Sum of loop elements
            long loopSum = loopStartElement;
            curr = loopStartElement;
            for (i = 1; i < loopSize; ++i) {
                curr = ((curr % m) * (curr % m)) % m;
                loopSum += curr;
            }
            long loops = (n - loopStart) / loopSize;
            sum += loops * loopSum;


            // Sum of remaining elements
            long rem = n - (loops * loopSize) - loopStart;
            if (rem > 0)
                sum += loopStartElement;
            curr = loopStartElement;
            for (i = 1; i < rem; ++i) {
                curr = ((curr % m) * (curr % m)) % m;
                sum += curr;
            }
            System.out.println(sum);
        }
    }
}
