package tests;
public class TestB {
    public static boolean test() // missing open brace
        String s = "";
        return s.isEmpty();
    }
    public static boolean test2() {
        return true;
    }
}