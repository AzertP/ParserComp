import java.util.*;
public class Main{
  public static void main(String[] args){
    Scanner sc = new Scanner(System.in);
    int N = sc.nextInt();
    long A[] = new long[N];
    for(int i=0;i<N;i++){
      A[i]=sc.nextLong();
    }
    Arrays.sort(A);
    long top =0;
    long nexttop =0;
    boolean isFindtop=false;
    for(int i=N-1;i>0;i--){
      if(A[i]==A[i-1]){
        top=A[i];
        isFindtop=!isFindtop;
        break;
      }
    }

    if(!isFindtop){
      System.out.println(0);
      return;
    }
    int cnt=0;
    for(int i=N-1;i>0;i--){
      if(A[i]==A[i-1]&&(A[i]!=top||cnt==2)){
        nexttop=A[i];
        System.out.println(top*nexttop);
        return;
      }else if(A[i]==A[i-1]){
        cnt++;
      }else{
        continue;
      }
    }
    System.out.println(0);
    
    
      
        
  }
  

  
}
