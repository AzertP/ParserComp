import java.io.*;

public class Main
{
  public static void main(String[] args) {
    try {
      BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
      int n = Integer.parseInt(br.readLine());
      String[] strArray = br.readLine().split(" ");
      int[] A = new int[n];
      for(int i=0; i<n; i++) {
        A[i] = Integer.parseInt(strArray[i]);
      }
      int totalCount = mergeSort(A, 0, n, 0);
      StringBuilder sb = new StringBuilder();
      for(int i=0; i<n-1; i++) {
        sb.append(A[i]).append(" ");
      }
      sb.append(A[n-1]);
      System.out.println(sb.toString());
      System.out.println(totalCount);
    } catch (IOException e) {
    }
  }

  public static int merge(int[] A, int left, int mid, int right, int count) {
    int n1 = mid - left;
    int n2 = right - mid;
    int[] L = new int[n1+1];
    int[] R = new int[n2+1];
    for(int i=0; i<n1; i++) {
      L[i] = A[left + i];
    }
    for(int i=0; i<n2; i++) {
      R[i] = A[mid + i];
    }
    L[n1] = Integer.MAX_VALUE;
    R[n2] = Integer.MAX_VALUE;

    int i = 0;
    int j = 0;

    for(int k = left; k<right; k++) {
      count += 1;
      if( L[i] <= R[j] ) {
        A[k] = L[i];
        i = i+1;
      } else {
        A[k] = R[j];
        j = j+1;
      }
    }

    return count;

  }

  public static int mergeSort(int[] A, int left, int right, int count) {
    int totalCount = count;
    if(left + 1 < right) {
      int mid = (left + right) / 2;
      totalCount += mergeSort(A, left, mid, count);
      totalCount += mergeSort(A, mid, right, count);
      totalCount += merge(A, left, mid, right, count);
    }
    return totalCount;
  }

}