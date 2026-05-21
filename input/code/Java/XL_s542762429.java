/**
 * Created at 22:17 on 2019/04/06
 */

import java.io.*;
import java.util.*;
import java.math.BigInteger;

public class Main {

  static FastScanner sc = new FastScanner();

  public static void main(String[] args) {

    int[] len = new int[3];
    for (int i=0; i<3; i++) {
      len[i] = sc.nextInt();
    }
    int K = sc.nextInt();
    long[][] val = new long[3][];
    val[0] = new long[len[0]];
    val[1] = new long[len[1]];
    val[2] = new long[len[2]];

    for (int t=0; t<3; t++) {
      for (int i=0; i<len[t]; i++) {
        val[t][i] = sc.nextLong();
      }
    }

    long[] X = f(val[0], val[1]);
    long[] Y = f(X, val[2]);

    for (int i=Y.length-1; i>=Y.length-K; i--) {
      System.out.println(Y[i]);
    }

  }

  static long[] f(long[] X, long[] Y) {
    long[] arr = new long[X.length*Y.length];
    for (int i=0; i<X.length; i++) {
      for (int j=0; j<Y.length; j++) {
        arr[i*Y.length+j] = X[i] + Y[j];
      }
    }
    Arrays.sort(arr);
    return Arrays.copyOfRange(arr, Math.max(0, arr.length-3000), arr.length);
  }

  static class Debugger {

    static long startTime;
    static boolean on = true;

    //入力受け取り後に呼び出す(入力時間をカウントしないため)
    static void start() {
      startTime = System.currentTimeMillis();
    }

    static void printTime() {
      if (on) System.out.println("(" + (System.currentTimeMillis() - startTime) + "ms)");
    }

    static void toggle() {
      on = !on;
    }

  }

  static class BigInt extends BigInteger {

    BigInteger MOD = new BigInteger("1000000007");
    static BigInt ZERO = new BigInt(0);
    static BigInt ONE = new BigInt(1);
    static BigInt TEN = new BigInt(10);

    public BigInt(long l) {
      super(Long.toString(l));
    }

    public BigInt(BigInteger i) {
      super(i.toByteArray());
    }

    public BigInt pow(int i) {
      return new BigInt(super.pow(i));
    }

    public BigInt[] divideAndRemainder(long l) {
      return divideAndRemainder(new BigInt(l));
    }

    public BigInt[] divideAndRemainder(BigInt i) {
      BigInt[] ret = new BigInt[2];
      BigInteger[] dAndR = super.divideAndRemainder(i);
      ret[0] = new BigInt(dAndR[0]);
      ret[1] = new BigInt(dAndR[1]);
      return ret;
    }

    public BigInt mod(long l) {
      return mod(new BigInt(l));
    }

    public BigInt mod(BigInt i) {
      return new BigInt(super.mod(i));
    }

    public BigInt mod() {
      return new BigInt(super.mod(MOD));
    }

    public BigInt add(long l) {
      return add(new BigInt(l));
    }

    public BigInt add(BigInt i) {
      return new BigInt(super.add(i));
    }

    public BigInt subtract(long l) {
      return subtract(new BigInt(l));
    }

    public BigInt subtract(BigInt i) {
      return new BigInt(super.subtract(i));
    }

    public BigInt multiply(long l) {
      return multiply(new BigInt(l));
    }

    public BigInt multiply(BigInt i) {
      return new BigInt(super.multiply(i));
    }

    public BigInt divide(long l) {
      return divide(new BigInt(l));
    }

    public BigInt divide(BigInt i) {
      return new BigInt(super.divide(i));
    }

    public BigInt abs() {
      return new BigInt(super.abs());
    }

  }

  /*
    add()でインデックスを指定しない場合指定されたソート順に違わない位置に追加する
    (ただしソートされていることが前提となる)
    Comparatorが0を返したとき、それらの順序は保証しない
    (TreeSet, TreeMapと違い削除はしない)
   */
  static class TreeList<E> extends ArrayList<E> {

    Comparator<? super E> comparator;

    TreeList(Comparator<? super E> c) {
      super();
      comparator = c;
    }

    /*
      ソート済みのリストに要素を追加する
     */
    public boolean add(E e) {
      int lowIndex = 0;
      int highIndex = size() - 1;
      int index = 0;

      if (size() == 0) {
        super.add(e);
        return true;
      }

      if (comparator.compare(e, get(0)) < 0) {
        index = 0;
      } else if (comparator.compare(e, get(highIndex)) > 0) {
        index = highIndex + 1;
      } else {
        while (lowIndex <= highIndex) {

          if (highIndex == lowIndex + 1 || highIndex == lowIndex) {
            index = highIndex;
            break;
          }

          int midIndex = (lowIndex + highIndex) / 2;
          ;

          if (comparator.compare(e, get(midIndex)) > 0) {
            lowIndex = midIndex;
          } else {
            highIndex = midIndex;
          }

        }
      }

      super.add(index, e);
      return true;
    }

  }

  static class FastScanner {
    private final InputStream in = System.in;
    private final byte[] buffer = new byte[1024];
    private int ptr = 0;
    private int buflen = 0;

    private boolean hasNextByte() {
      if (ptr < buflen) {
        return true;
      } else {
        ptr = 0;
        try {
          buflen = in.read(buffer);
        } catch (IOException e) {
          e.printStackTrace();
        }
        if (buflen <= 0) {
          return false;
        }
      }
      return true;
    }

    private int readByte() {
      if (hasNextByte()) return buffer[ptr++];
      else return -1;
    }

    private static boolean isPrintableChar(int c) {
      return 33 <= c && c <= 126;
    }

    private void skipUnprintable() {
      while (hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++;
    }

    public boolean hasNext() {
      skipUnprintable();
      return hasNextByte();
    }

    public String next() {
      if (!hasNext()) throw new NoSuchElementException();
      StringBuilder sb = new StringBuilder();
      int b = readByte();
      while (isPrintableChar(b)) {
        sb.appendCodePoint(b);
        b = readByte();
      }
      return sb.toString();
    }

    public long nextLong() {
      if (!hasNext()) throw new NoSuchElementException();
      long n = 0;
      boolean minus = false;
      int b = readByte();
      if (b == '-') {
        minus = true;
        b = readByte();
      }
      if (b < '0' || '9' < b) {
        throw new NumberFormatException();
      }
      while (true) {
        if ('0' <= b && b <= '9') {
          n *= 10;
          n += b - '0';
        } else if (b == -1 || !isPrintableChar(b)) {
          return minus ? -n : n;
        } else {
          throw new NumberFormatException();
        }
        b = readByte();
      }
    }

    public int nextInt() {
      return (int) nextLong();
    }
  }

}
