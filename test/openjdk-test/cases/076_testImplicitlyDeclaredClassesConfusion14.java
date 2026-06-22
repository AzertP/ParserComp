package tests;
public class TestB {
    public static boolean test() // missing open brace
        String s = "";
        s.length();
        if (true); //force parse as block
    public static boolean test2() {
        return true;
    }
}