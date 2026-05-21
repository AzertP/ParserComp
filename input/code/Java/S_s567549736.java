import java.util.Scanner;
	public class Main{
		public static void main(String[]args){
			Scanner sc = new Scanner(System.in);
			int h;
			for(h=1;;h++){
				int x=sc.nextInt();
				int y=sc.nextInt();
				int i,j;
				if(x!=0&&y!=0){
					for(i=0;i<x;i++){
						for(j=0;j<y;j++){
							System.out.printf("#");
						}
						System.out.println("");
					}
					System.out.println("");
					}else{
						break;
					}
			}
		}
	}