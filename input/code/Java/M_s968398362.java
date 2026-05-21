import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		List<Integer> height = new ArrayList<>();
		List<Integer> width = new ArrayList<>();
		int h = 0;
		int w = 0;


		int count = 0;
		boolean judge = false;
		while(!judge){
			h = sc.nextInt();
			w = (sc.nextInt());

			if(h == 0 && w == 0){
				judge = true;
			}else{
				height.add(h);
				width.add(w);
				count++;
			}
		}

		for(int i = 0; i < count; i++){
			for(int j = 0; j < height.get(i); j++){
				for(int k = 0; k < width.get(i); k++){
					if(j == 0 || j == (height.get(i) - 1)){
						System.out.print("#");
					}else{
						System.out.print("#");
						for(int l = 0; l < width.get(i) - 2; l++){
							System.out.print(".");
						}
						System.out.print("#");
						break;
					}
				}
				System.out.println();
			}
			System.out.println();
		}

	}

}

