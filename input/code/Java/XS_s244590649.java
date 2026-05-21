import java.util.*;

class Main{
  public static void main(String args[]){
    Scanner inputs = new Scanner(System.in);
    int seconds = inputs.nextInt();
    int minutes = seconds / 60;
    seconds = seconds % 60;
    int hours = minutes / 60;
    minutes = minutes % 60;
    System.out.println(hours + ":" + minutes + ":" + seconds);
  }
}