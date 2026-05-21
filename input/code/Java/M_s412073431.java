import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringBuilder sb = new StringBuilder();
        while (true){
            String line = br.readLine();
            String[] ln = line.split(" ");
            int H = Integer.parseInt(ln[0]);
            int W = Integer.parseInt(ln[1]);
            if (H==0 && W==0) break;
            for (int j=0; j<H; j++){
                if (j==0|j==H-1){
                    for (int i=0; i<W; i++){
                        sb.append("#");
                    }
                }else{
                    for (int i=0; i<W; i++){
                        if (i==0|i==W-1){
                            sb.append("#");
                        }else{
                            sb.append(".");
                        }
                    }
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
        System.out.print(sb);
    }
}