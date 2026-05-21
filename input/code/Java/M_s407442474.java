import java.util.*;

 // Compiler version JDK 11.0.2

 public class Main
 {
   public static void main(String args[])
   { 
    Scanner sc= new Scanner(System.in);
    int a = sc.nextInt();
    int b = sc.nextInt();
    int c = sc.nextInt();
    int d = sc.nextInt();
    
    
    while(a>0 && c>0)
    {
      
      c=c-b;
      if(c>0)
      
      a=a-d;
      
      
    }
    if(a>0)
    System.out.println("Yes");
    else
    System.out.println("No");
   }
 }
