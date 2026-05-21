import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();

	    List<Stock> stockList = new ArrayList<Stock>();

		for(int i = 0; i < n; i++) {
			int v = sc.nextInt();
			int w = sc.nextInt();
			int t = (int)sc.next().charAt(0);
			long d = sc.nextLong();
			String s = sc.next();

			Stock st = new Stock(v, w, t, d, s);

			stockList.add(st);
		}

	    Comparator<Stock> comparator = Comparator.comparing(Stock::getValue)
	    		.thenComparing(Stock::getWeight)
	    		.thenComparing(Stock::getType)
	    		.thenComparing(Stock::getDate)
	    		.thenComparing(Stock::getName);

		Collections.sort(stockList, comparator);

		for(int i = 0; i < n; i++) {
			System.out.println(stockList.get(i).getValue() + " " + stockList.get(i).getWeight() + " "
		+ (char)stockList.get(i).getType() + " " + stockList.get(i).getDate() + " " + stockList.get(i).getName());
		}
	}
}

class Stock {
	private int value;
	private int weight;
	private int type;
	private long date;
	private String name;

	Stock(int a, int b, int c, long d, String s) {
		this.value = a;
		this.weight = b;
		this.type = c;
		this.date = d;
		this.name = s;
	}

	public int getValue() {
		return this.value;
	}

	public int getWeight() {
		return this.weight;
	}

	public int getType() {
		return this.type;
	}

	public long getDate() {
		return this.date;
	}

	public String getName() {
		return this.name;
	}
}
