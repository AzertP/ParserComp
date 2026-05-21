import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Lesson7C 表の行数rと列数c、r × c の要素を持つ表を読み込んで、
 * 各行と列の合計を挿入した新しい表を出力するプログラム
 *
 * @author Internship
 *
 */
public class Main{

	/**
	 * プログラムのエントリポイント
	 *
	 * @param args
	 */
	public static void main(String[] args) {


		//行や列の各数字の一時保管場所
		int intVector = 0;

		//入力値の各行の文字列の一時保管場所
		String strLineAll = "";

		//行の合計値
		int intColumnSum = 0;

		//行と列の全合計値
		int intSum = 0;

		//一行目を読み込む
		try {
			BufferedReader insBR = new BufferedReader(new InputStreamReader(System.in));

			String strLine1 = insBR.readLine();

			//ばらけさせる
			String[] strAryLine1 = strLine1.split(" ");

			//行の数
			int intLine = Integer.parseInt(strAryLine1[0]);

			//列の数
			int intColumn = Integer.parseInt(strAryLine1[1]);

			//行と列が決まったところで配列の大きさが決まる
			int[][] intTable = new int[intLine + 1][intColumn + 1];

			//二行目以降を読み込む
			//i行目
			for(int i = 0; i < intLine; i++) {
				strLineAll = insBR.readLine();

			//ばらけさせる
				String[] strAryLine = strLineAll.split(" ");

				//int型にキャスト
				//j列目
				for(int j = 0; j < intColumn; j++) {
					intVector = Integer.parseInt(strAryLine[j]);

					//配列に格納
					//i行のj列
					intTable[i][j] = intVector;

					//行の合計値をカウント
					intColumnSum += intVector;

					//列の合計値をカウント
					intTable[intLine][j] += intVector;

					//総合計値をカウント
					intSum += intVector;
				}
				//行の合計値を代入
				intTable[i][intColumn] = intColumnSum;

				//1行分終わったのでリセット
				intColumnSum = 0;
			}

			//総合計値を代入
			intTable[intLine][intColumn] = intSum;

			//計算して出力
			for(int i = 0; i <= intLine; i++) {
				for(int j = 0; j <= intColumn; j++) {

					if(j == 0) {
						System.out.print(intTable[i][j]);
					}else {
						System.out.print(" " + intTable[i][j]);
					}
				}
				System.out.println("");
			}

		} catch (IOException e) {
			System.out.println(e);
		} catch (NumberFormatException e) {
			System.out.println("数字で入力して下さい");
		}
	}
}