package tests;
public class TestB {
    public static boolean test() // missing open brace
        final String s = "";
        return s.isEmpty();
    }
    public static boolean test2() {
        return true;
    }
}