import java.io.*;
import java.util.*;

public class Main {

    public static ArrayList<Integer> r;
    public static ArrayList<Integer> g;
    public static ArrayList<Integer> b;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        scanner.nextLine();
        String str = scanner.nextLine();
        char[] s = str.toCharArray();
        r = new ArrayList<>();
        g = new ArrayList<>();
        b = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (s[i] == 'R') {
                r.add(i);
            }
            if (s[i] == 'G') {
                g.add(i);
            }
            if (s[i] == 'B') {
                b.add(i);
            }
        }
        long ans = 0;
        for (Integer val1 : b) {
            for (Integer val2 : r) {
                if (val2 > val1) {
                    ans += startG(val2) - startG(val1);
                    if ((val2 + val1) % 2 == 0) {
                        if (contG((val2 + val1) / 2)) {
                            ans -= 1;
                        }
                    }
                }
            }
        }

        for (Integer val1 : r) {
            for (Integer val2 : b) {
                if (val2 > val1) {
                    ans += startG(val2) - startG(val1);
                    if ((val2 + val1) % 2 == 0) {
                        if (contG((val2 + val1) / 2)) {
                            ans -= 1;
                        }
                    }
                }
            }
        }

        for (Integer val1 : g) {
            for (Integer val2 : r) {
                if (val2 > val1) {
                    ans += startB(val2) - startB(val1);
                    if ((val2 + val1) % 2 == 0) {
                        if (contB((val2 + val1) / 2)) {
                            ans -= 1;
                        }
                    }
                }
            }
        }

        for (Integer val1 : r) {
            for (Integer val2 : g) {
                if (val2 > val1) {
                    ans += startB(val2) - startB(val1);
                    if ((val2 + val1) % 2 == 0) {
                        if (contB((val2 + val1) / 2)) {
                            ans -= 1;
                        }
                    }
                }
            }
        }

        for (Integer val1 : b) {
            for (Integer val2 : g) {
                if (val2 > val1) {
                    ans += startR(val2) - startR(val1);
                    if ((val2 + val1) % 2 == 0) {
                        if (contR((val2 + val1) / 2)) {
                            ans -= 1;
                        }
                    }
                }
            }
        }

        for (Integer val1 : g) {
            for (Integer val2 : b) {
                if (val2 > val1) {
                    ans += startR(val2) - startR(val1);
                    if ((val2 + val1) % 2 == 0) {
                        if (contR((val2 + val1) / 2)) {
                            ans -= 1;
                        }
                    }
                }
            }
        }
        System.out.println(ans);
    }

    private static int startB(int n) {
        if (b.size() == 0) {
            return 0;
        }
        int right = b.size();
        int left = 0;
        while (right - left > 1) {
            int m = (right + left) / 2;
            if (b.get(m) >= n) {
                right = m;
            } else {
                left = m;
            }
        }
        if (b.get(left) > n) {
            return left;
        }
//        if (b.get(right) >= n) {
//            return right + 1;
//        }
        return left + 1;
    }

    private static boolean contB(int n) {
        if (b.size() == 0) {
            return false;
        }
        int right = b.size() - 1;
        int left = 0;
        while (right - left > 1) {
            int m = (right + left) / 2;
            if (b.get(m) >= n) {
                right = m;
            } else {
                left = m;
            }
        }
        if (b.get(left) == n) {
            return true;
        }
        return b.get(right) == n;
    }

    private static int startR(int n) {
        if (r.size() == 0) {
            return 0;
        }
        int right = r.size();
        int left = 0;
        while (right - left > 1) {
            int m = (right + left) / 2;
            if (r.get(m) >= n) {
                right = m;
            } else {
                left = m;
            }
        }
        if (r.get(left) > n) {
            return left;
        }
//        if (r.get(right) >= n) {
//            return right + 1;
//        }
        return left + 1;
    }

    private static boolean contR(int n) {
        if (r.size() == 0) {
            return false;
        }
        int right = r.size() - 1;
        int left = 0;
        while (right - left > 1) {
            int m = (right + left) / 2;
            if (r.get(m) >= n) {
                right = m;
            } else {
                left = m;
            }
        }
        if (r.get(left) == n) {
            return true;
        }
        return r.get(right) == n;
    }

    private static int startG(int n) {
        if (g.size() == 0) {
            return 0;
        }
        int right = g.size();
        int left = 0;
        while (right - left > 1) {
            int m = (right + left) / 2;
            if (g.get(m) >= n) {
                right = m;
            } else {
                left = m;
            }
        }
        if (g.get(left) > n) {
            return left;
        }
//        if (g.get(right) >= n) {
//            return right + 1;
//        }
        return left + 1;
    }

    private static boolean contG(int n) {
        if (g.size() == 0) {
            return false;
        }
        int right = g.size() - 1;
        int left = 0;
        while (right - left > 1) {
            int m = (right + left) / 2;
            if (g.get(m) >= n) {
                right = m;
            } else {
                left = m;
            }
        }
        if (g.get(left) == n) {
            return true;
        }
        return g.get(right) == n;
    }
}
