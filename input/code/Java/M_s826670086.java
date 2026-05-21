import java.util.Scanner;
import java.util.stream.Stream;

public class Main {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		long[] line = Stream.of(sc.nextLine().split(" ", 0)).mapToLong(Long::parseLong).toArray();
		long[] desk_a = Stream.of(sc.nextLine().split(" ", 0)).mapToLong(Long::parseLong).toArray();
		long[] desk_b = Stream.of(sc.nextLine().split(" ", 0)).mapToLong(Long::parseLong).toArray();

		long desk_a_sum = 0;
		long desk_b_sum = 0;
		long max = 0;
		int piv = -1;
		for (int i = 0; i < line[1]; i++) {
			if (desk_b_sum + desk_b[i] > line[2]) {
				break;
			}
			desk_b_sum += desk_b[i];
			max++;
			piv++;
		}
		long max_let = max;
		for (int i = 0; i < line[0]; i++) {
			max_let++;
			desk_a_sum += desk_a[i];
			if (piv > -1) {
				for (int j = piv; j >= 0; j--) {
					if (desk_b_sum <= line[2] - desk_a_sum ) {
						if (max < max_let) {
							max = max_let;
						}
						break;
					} else {
						desk_b_sum -= desk_b[j];
						max_let--;
						piv--;
					}
				}
			} else {
				if (desk_b_sum + desk_a_sum <= line[2] ) {
					if (max < max_let) {
						max = max_let;
					}
				}else {
					break;
				}
			}
		}

		System.out.println(max);
	}
}
