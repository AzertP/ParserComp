package t;
class Test {
    private int t(Integer i, boolean b) {
        switch (i) {
            case 0 when b -> {}
            case null when b -> {}
            default when b -> {}
        }
        return switch (i) {
            case 0 when b -> 0;
            case null when b -> 0;
            default when b -> 0;
        };
    }
}