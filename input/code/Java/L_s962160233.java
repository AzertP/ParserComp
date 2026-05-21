import java.awt.geom.Point2D;
import java.awt.geom.Point2D.Double;
import java.util.*;

public class Main {
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		while(true){
			int n = sc.nextInt();
			if(n==0) break;
			
			Point2D.Double[] p = new Point2D.Double[n];
			for(int i=0;i<n;i++) p[i] = new Point2D.Double(sc.nextDouble(), sc.nextDouble());
			
			Arrays.sort(p, new Comparator<Point2D.Double>(){
				public int compare(Point2D.Double o1, Point2D.Double o2) {
					if(o1.y-o2.y>0) return 1;
					else return -1;
				}
			});
			
			int max = 1;
			for(int i=0;i<n;i++){
				for(int j=i+1;j<n;j++){
					if(p[j].y-p[i].y>2.0001) break;
					if(p[j].distance(p[i])>2.0001) continue;
					
					double a, b, c, A, B, C;
					double xa = p[i].x;
					double ya = p[i].y;
					double xb = p[j].x;
					double yb = p[j].y;
					double[] x = new double[2];
					double[] y = new double[2];
					
					A = xa-xb;
					B = ya-yb;
					C = (A*xa+A*xb+B*ya+B*yb)/2;
					if(A==0){
						c = (C/B-ya)*(C/B-ya)+xa*xa-1;
						x[0] = xa+Math.sqrt(xa*xa-c);
						x[1] = xa-Math.sqrt(xa*xa-c);
						y[0] = C/B;
						y[1] = C/B;
					}else if(B==0){
						c = (C/A-xa)*(C/A-xa)+ya*ya-1;
						y[0] = ya+Math.sqrt(ya*ya-c);
						y[1] = ya-Math.sqrt(ya*ya-c);
						x[0] = C/A;
						x[1] = C/A;
					}else{
						a = 1+(A*A)/(B*B);
						b = (2*A*ya*B-2*A*C-2*xa*B*B)/(B*B);
						c = (C/B-ya)*(C/B-ya)+xa*xa-1;
						x[0] = (-b+Math.sqrt(b*b-4*a*c))/(2*a);
						x[1] = (-b-Math.sqrt(b*b-4*a*c))/(2*a);
						y[0] = (C-A*x[0])/B;
						y[1] = (C-A*x[1])/B;
					}
					
					for(int d=0;d<2;d++){
						int cnt = 0;
						for(int e=0;e<n;e++){
							if(p[e].y-y[d]>1.0001) break;
							if(p[e].distance(x[d], y[d])<1.0001) cnt++;
						}
						max = Math.max(max, cnt);
					}
				}
			}
			
			System.out.println(max);
		}	
	}	
}