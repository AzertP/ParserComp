import java.util.Scanner;

public class Main{
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		double[] x = new double[3];
		double[] y = new double[3];
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < 3; i++) {
				x[i] = sc.nextDouble();
				y[i] = sc.nextDouble();
			}
			double S = ((x[0] - x[2]) * (y[1] - y[2]) - (x[1] - x[2]) * (y[0] - y[2]));
			S = Math.abs(S / 2.0);
			double A = (x[1] - x[2])* (x[1] - x[2]) + (y[1] - y[2])* (y[1] - y[2]);
			double B = (x[0] - x[2])* (x[0] - x[2]) + (y[0] - y[2])* (y[0] - y[2]);
			double C = (x[1] - x[0])* (x[1] - x[0]) + (y[1] - y[0])* (y[1] - y[0]);
			double px = (A*(B+C-A)*x[0] + B*(C+A-B)*x[1] + C*(A+B-C)*x[2])/(16.0*S*S);
			double py = (A*(B+C-A)*y[0] + B*(C+A-B)*y[1] + C*(A+B-C)*y[2])/(16.0*S*S);
			double r = (px - x[0]) * (px - x[0]) + (py - y[0]) * (py - y[0]);
			r = Math.sqrt(r);
			System.out.printf("%.3f %.3f %.3f\n", px, py, r);
		}
		sc.close();
	}
}
