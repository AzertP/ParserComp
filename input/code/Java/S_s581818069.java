import java.util.*;
public class Main{
	public static void main(String[] args){
    	Scanner sc = new Scanner(System.in);
      int x = sc.nextInt();
      int y = sc.nextInt();
      int[] array1 = {1,3,5,7,8,10,12};
      int[] array2 = {4,6,9,11,0,0,0};
      int count1 = 0;
      int count2 = 0;
      for(int i = 0;i < 7;i++){
      	if(x == array1[i]||y == array1[i])count1++;
        if(x == array2[i]||y == array2[i])count2++;
      }
      if(count1 == 2||count2 == 2)System.out.println("Yes");
      else System.out.println("No");
      
    }
}
