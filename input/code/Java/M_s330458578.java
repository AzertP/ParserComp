import java.util.Scanner;

class Main {

	public static void main(String[] args) {

		Scanner sc = null;
		try {
			sc = new Scanner(System.in);
			int countRectangle = 0;
			int countRhombus = 0;
			while (sc.hasNextLine()) {
				String[] input = sc.nextLine().split(",");
				int[] side = new int[3];

				for (int i = 0; i < side.length; i++) {
					side[i] = Integer.parseInt(input[i]);
				}

				if (isRectangle(side)) {
					countRectangle++;
				} else if (isRhombus(side))
					countRhombus++;
			}

			System.out.println(countRectangle);
			System.out.println(countRhombus);

		} finally {
			sc.close();
		}
	}

	public static boolean isRectangle(int[] side) {
		if (Math.pow(side[0], 2.0) + Math.pow(side[1], 2.0) == Math.pow(
				side[2], 2.0)) {
			return true;
		}
		return false;
	}

	public static boolean isRhombus(int[] side) {
		if (side[0] == side[1]) {
			return true;
		}
		return false;
	}
}