package t;
class Test {
    void test1() {
        try (AutoCloseable ac = null) {}
    }
    void test2() {
        try (@Ann AutoCloseable withAnnotation = null) {}
    }
    void test3() {
        try (final AutoCloseable withFinal = null) {}
    }
    void test4() {
        try (final @Ann AutoCloseable withAnnotationFinal = null) {}
    }
    @interface Ann {}
}
