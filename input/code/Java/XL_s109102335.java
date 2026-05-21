import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Main {
	final int _intMax = Integer.MAX_VALUE; // =2147483647>10^9
	final int _intMin = Integer.MIN_VALUE;
	final long _longMax = Long.MAX_VALUE; // =9223372036854775807L>10^18
	final long _longMin = Long.MIN_VALUE;
	final char[] _azAry = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
			's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };

	void solve() {
		int n = readNum();
		int[] ia = new int[4];
		for (int i = 0; i < 4; i++)
			ia[i] = reed(n, 3 - i);
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					int ans = 0;
					if (i == 0)
						ans = ia[0] + ia[1];
					else
						ans = ia[0] - ia[1];
					if (j == 0)
						ans += ia[2];
					else
						ans -= ia[2];
					if (k == 0)
						ans += ia[3];
					else
						ans -= ia[3];
					if (ans == 7) {
						out(ia, i, j, k);
						return;
					}
				}
			}
		}
	}

	void out(int[] ia, int i, int j, int k) {
		p(ia[0]);
		p(i == 0 ? '+' : '-');
		p(ia[1]);
		p(j == 0 ? '+' : '-');
		p(ia[2]);
		p(k == 0 ? '+' : '-');
		p(ia[3]);
		pln("=7");
	}

	// -----------------------------------------------------
	// 2018/05/05 r25
	// -----------------------------------------------------
	List<Character> getazList() {
		List<Character> list = new ArrayList<>();
		for (char ch : _azAry)
			list.add(ch);
		return list;
	}

	int getDx(int idx) {
		int[] dx = { 0, 1, 1, 1, 0, -1, -1, -1 };
		return dx[idx];
	}

	int getDy(int idx) {
		int[] dy = { -1, -1, 0, 1, 1, 1, 0, -1 };
		return dy[idx];
	}

	class Bitmap {
		int mx;
		int my;
		boolean[][] map;

		public Bitmap(int mx, int my) {
			this.mx = mx;
			this.my = my;
			map = new boolean[my + 2][mx + 2];
		}

		public boolean is(int x, int y) {
			return map[y][x];
		}

		public void set(int x, int y, boolean b) {
			map[y][x] = b;
		}
	}

	class CharList implements Iterable<Character> {
		class CharComparator implements Comparator<Character> {
			int sign;

			public CharComparator(boolean bAsc) {
				sign = bAsc ? 1 : -1;
			}

			public int compare(Character o1, Character o2) {
				return sign * Character.compare(o1, o2);
			}
		}

		List<Character> list = new ArrayList<>();
		CharComparator asc = new CharComparator(true);
		CharComparator desc = new CharComparator(false);

		public void add(char ch) {
			list.add(ch);
		}

		public char get(int idx) {
			return list.get(idx);
		}

		public char getLast() {
			return list.get(list.size() - 1);
		}

		public Iterator<Character> iterator() {
			return list.iterator();
		}

		public void remove(int idx) {
			list.remove(idx);
		}

		public void removeLast() {
			list.remove(list.size() - 1);
		}

		public int size() {
			return list.size();
		}

		public void sort(boolean bAsc) {
			if (bAsc)
				Collections.sort(list, asc);
			else
				Collections.sort(list, desc);
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (char ch : list)
				sb.append(ch);
			return sb.toString();
		}
	}

	class Counter<K> {
		Map<K, Integer> map = new HashMap<>();

		public Counter() {
		}

		public Counter(Iterable<K> itr) {
			for (K obj : itr) {
				add(obj);
			}
		}

		public void add(K key) {
			Integer cnt = map.get(key);
			if (cnt == null)
				map.put(key, 1);
			else
				map.put(key, cnt + 1);
		}

		public int get(K key) {
			Integer cnt = map.get(key);
			if (cnt == null)
				return 0;
			else
				return cnt;
		}

		public Set<K> keySet() {
			return map.keySet();
		}
	}

	class IntList implements Iterable<Integer> {
		class Info {
			int idx;
			int val;

			public Info(int idx, int val) {
				this.idx = idx;
				this.val = val;
			}

			public String toString() {
				return "(" + idx + ", " + val + ")";
			}
		}

		class InfoComparator implements Comparator<Info> {
			int sign;

			public InfoComparator(boolean bAsc) {
				sign = bAsc ? 1 : -1;
			}

			public int compare(Info o1, Info o2) {
				return sign * Integer.compare(o1.val, o2.val);
			}
		}

		List<Info> list = new ArrayList<>();
		InfoComparator asc = new InfoComparator(true);
		InfoComparator desc = new InfoComparator(false);

		public IntList() {
		}

		public IntList(int[] ia) {
			for (int i = 0; i < ia.length; i++)
				add(ia[i]);
		}

		public void add(int val) {
			list.add(new Info(list.size(), val));
		}

		public void add(int idx, int val) {
			list.add(new Info(idx, val));
		}

		public int getIdx(int idx) {
			return list.get(idx).idx;
		}

		public int getVal(int idx) {
			return list.get(idx).val;
		}

		public int getLastVal() {
			return list.get(list.size() - 1).val;
		}

		public Iterator<Integer> iterator() {
			List<Integer> vallist = new ArrayList<>();
			for (Info info : list)
				vallist.add(info.val);
			return vallist.iterator();
		}

		public void remove(int idx) {
			list.remove(idx);
		}

		public void removeLast() {
			list.remove(list.size() - 1);
		}

		public int size() {
			return list.size();
		}

		public void sort(boolean bAsc) {
			if (bAsc)
				Collections.sort(list, asc);
			else
				Collections.sort(list, desc);
		}
	}

	class Point {
		int x;
		int y;

		public Point(int x, int y) {
			this.x = x;
			this.y = y;
		}

		public Point(Point pt) {
			this.x = pt.x;
			this.y = pt.y;
		}

		public boolean equals(Object o) {
			if (o instanceof Point) {
				Point that = (Point) o;
				return (x == that.x) && (y == that.y);
			}
			return false;
		}

		public long getManhattanDistance(Point pt) {
			return abs((long) pt.x - this.x) + abs((long) pt.y - this.y);
		}

		public int hashCode() {
			return x + (y * 31);
		}

		public String toString() {
			return "(" + x + ", " + y + ")";
		}
	}

	class PointList implements Iterable<Point> {
		class PointComparator implements Comparator<Point> {
			int mode;

			public PointComparator(int prop, boolean bAsc) {
				switch (prop) {
				case 1:
					if (bAsc)
						this.mode = 11;
					else
						this.mode = 12;
					break;
				case 2:
					if (bAsc)
						this.mode = 21;
					else
						this.mode = 22;
					break;
				default:
					throw new RuntimeException();
				}
			}

			public int compare(Point p1, Point p2) {
				switch (mode) {
				case 11:
					return 1 * Integer.compare(p1.x, p2.x);
				case 12:
					return -1 * Integer.compare(p1.x, p2.x);
				case 21:
					return 1 * Integer.compare(p1.y, p2.y);
				case 22:
					return -1 * Integer.compare(p1.y, p2.y);
				}
				throw new IllegalStateException();
			}
		}

		List<Point> list = new ArrayList<>();

		public void add(int x, int y) {
			list.add(new Point(x, y));
		}

		public Point get(int idx) {
			return list.get(idx);
		}

		public Iterator<Point> iterator() {
			return list.iterator();
		}

		public void remove(int idx) {
			list.remove(idx);
		}

		public int size() {
			return list.size();
		}

		public void sort(int prop, boolean bAsc) {
			PointComparator c = new PointComparator(prop, bAsc);
			Collections.sort(list, c);
		}
	}

	class StrList implements Iterable<String> {
		List<String> list = new ArrayList<>();

		public StrList() {
		}

		public StrList(String[] sa) {
			for (int i = 0; i < sa.length; i++)
				list.add(sa[i]);
		}

		public Iterator<String> iterator() {
			return list.iterator();
		}
	}

	int abs(int a) {
		return (a >= 0) ? a : -a;
	}

	long abs(long a) {
		return (a >= 0) ? a : -a;
	}

	long ceil(long a, long b) {
		if (a < 0) {
			return -1 * floor(-a, b);
		}
		return ((a + b - 1) / b) * b;
	}

	long floor(long a, long b) {
		if (a < 0) {
			return -1 * ceil(-a, b);
		}
		return (a / b) * b;
	}

	int max(int a, int b) {
		return (a > b) ? a : b;
	}

	long max(long a, long b) {
		return (a > b) ? a : b;
	}

	int min(int a, int b) {
		return (a < b) ? a : b;
	}

	long min(long a, long b) {
		return (a < b) ? a : b;
	}

	int reed(long a, int n) {
		while (n-- > 0)
			a /= 10;
		return (int) (a % 10);
	}

	int sqrt(long a) {
		return (int) Math.sqrt(a);
	}

	int pint(String s) {
		return Integer.parseInt(s);
	}

	long plong(String s) {
		return Long.parseLong(s);
	}

	String readLine() {
		try {
			return _in.readLine();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	int readNum() {
		String line = readLine();
		return pint(line);
	}

	long readLong() {
		String line = readLine();
		return plong(line);
	}

	String[] readFlds() {
		String line = readLine();
		return line.split(" ");
	}

	String[] readLines(int n) {
		String[] lines = new String[n];
		for (int i = 0; i < n; i++)
			lines[i] = readLine();
		return lines;
	}

	CharList readChars() {
		CharList list = new CharList();
		String line = readLine();
		for (int i = 0; i < line.length(); i++)
			list.add(line.charAt(i));
		return list;
	}

	int[] readNums() {
		String[] flds = readFlds();
		int[] nums = new int[flds.length];
		for (int i = 0; i < flds.length; i++)
			nums[i] = pint(flds[i]);
		return nums;
	}

	int[] readNums(int n) {
		int[] nums = new int[n];
		for (int i = 0; i < n; i++)
			nums[i] = readNum();
		return nums;
	}

	long[] readLongs() {
		String[] flds = readFlds();
		long[] nums = new long[flds.length];
		for (int i = 0; i < flds.length; i++)
			nums[i] = plong(flds[i]);
		return nums;
	}

	long[] readLongs(int n) {
		long[] nums = new long[n];
		for (int i = 0; i < n; i++)
			nums[i] = readLong();
		return nums;
	}

	Main pln() {
		_out.println();
		return this;
	}

	Main p(char c) {
		_out.print(c);
		return this;
	}

	Main p(char c, int n) {
		for (int i = 0; i < n; i++)
			p(c);
		return this;
	}

	Main pln(char c) {
		_out.println(c);
		return this;
	}

	Main p(double d) {
		_out.print(d);
		return this;
	}

	Main pln(double d) {
		_out.println(d);
		return this;
	}

	Main p(long l) {
		_out.print(l);
		return this;
	}

	Main pln(long l) {
		_out.println(l);
		return this;
	}

	Main p(String s) {
		_out.print(s);
		return this;
	}

	Main p(String s, int idx) {
		_out.print(s.charAt(idx));
		return this;
	}

	Main pln(String s) {
		_out.println(s);
		return this;
	}

	Main pln(int[] ia) {
		for (int i = 0; i < ia.length; i++)
			_out.println(ia[i]);
		return this;
	}

	Main pln(long[] la) {
		for (int i = 0; i < la.length; i++)
			_out.println(la[i]);
		return this;
	}

	static BufferedReader _in;
	static PrintWriter _out;
	static boolean _bElapsed = false;

	public static void main(String[] args) {
		long start = System.currentTimeMillis();
		_in = new BufferedReader(new InputStreamReader(System.in));
		_out = new PrintWriter(System.out);
		new Main().solve();
		_out.flush();
		long end = System.currentTimeMillis();
		if (_bElapsed)
			System.err.println((end - start) + "ms");
	}
}
