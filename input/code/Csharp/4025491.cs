using System;
public class Hello{
    public static void Main(){
        // Your code here!
        int n = int.Parse(Console.ReadLine());
        string []s = Console.ReadLine().Split();
        int[] A = new int[n];
        for(int i=0;i<n;++i){
            A[i] = int.Parse(s[i]);
        }
        for(int j=0;j<n;++j){
            int tmp = A[j];
            int i = j - 1;
            while(i>=0&&A[i]>tmp){
                A[i+1] = A[i];
                --i;
            }
            A[i+1] = tmp;
            for(i=0;i<n;++i){
                if(i!=0)Console.Write(" ");
                Console.Write(A[i]);
            }
            Console.Write("\n");
        }
    }
}

