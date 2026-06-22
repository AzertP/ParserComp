package t;
class Test {
    private void test1(int i) {
        switch (i) {
            default:
        }
    }
    private void test2(int i) {
        switch (i) {
            case default:
        }
    }
    private int test3(int i) {
        return switch (i) {
            default: yield 0;
        }
    }
    private int test4(int i) {
        return switch (i) {
            case default: yield 0;
        }
    }
    private void test5(int i) {
        switch (i) {
            default -> {}
        }
    }
    private void test6(int i) {
        switch (i) {
            case default -> {}
        }
    }
    private int test5(int i) {
        return switch (i) {
            default -> { yield 0; }
        }
    }
    private int test6(int i) {
        return switch (i) {
            case default -> { yield 0; }
        }
    }
    private int test7(int i) {
        return switch (i) {
            default -> 0;
        }
    }
    private int test8(int i) {
        return switch (i) {
            case default -> 0;
        }
    }
}
