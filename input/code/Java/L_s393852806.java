import java.util.*;
import java.math.*;
import java.awt.geom.*;
import java.io.*;
   
   
public class Main {
	static int INF = 2 << 28;
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(true) {
			int w = sc.nextInt();
			int h = sc.nextInt();
			if(w == 0 && h == 0) break;
			int f = sc.nextInt();
			int m = sc.nextInt();
			int o = sc.nextInt();
			
			int[][] map = new int[h][w];
			for(int i = 0; i < h; i++) {
				for(int j = 0; j < w; j++) {
					map[i][j] = sc.nextInt();
				}
			}
			
			int[][][] memo = new int[h][w][m+1];
			for(int i = 0; i < h; i++) {
				for(int j = 0; j < w; j++) {
					Arrays.fill(memo[i][j], INF);
				}
			}
			PriorityQueue<Data> queue = new PriorityQueue<Data>();
			for(int i = 0; i < w; i++) {
				boolean[] tmp = new boolean[w];
				if(map[0][i] > 0 && o > 1) {
					queue.add(new Data(0,i,0,Math.min(m, o - 1 + map[0][i]),tmp));
				}
				else  {
					queue.add(new Data(0,i,-map[0][i],o - 1,tmp));
				}
			}
			
			int ans = Integer.MAX_VALUE;
			while(!queue.isEmpty()) {
				Data tmp = queue.poll();
				
				if(tmp.o <= 0) continue;
				tmp.al[tmp.x] = true;
				
				if(tmp.y == h-1) {
					ans = Math.min(ans, tmp.c);
				}
				
				if(tmp.c >= memo[tmp.y][tmp.x][tmp.o]) continue;
				memo[tmp.y][tmp.x][tmp.o] = tmp.c;
				
				if(tmp.x != w-1) {
					if(tmp.al[tmp.x+1]) {
						    queue.add(new Data(tmp.y,tmp.x+1,tmp.c                      ,tmp.o-1                                     ,Arrays.copyOf(tmp.al,tmp.al.length)));
					}
					else {
						if(map[tmp.y][tmp.x+1] > 0 && tmp.o != 1) {
							queue.add(new Data(tmp.y,tmp.x+1,tmp.c                      ,Math.min(m, tmp.o - 1 + map[tmp.y][tmp.x+1]),Arrays.copyOf(tmp.al,tmp.al.length)));
						}
						else if(map[tmp.y][tmp.x+1] < 0){
							queue.add(new Data(tmp.y,tmp.x+1,tmp.c - map[tmp.y][tmp.x+1],tmp.o-1                                     ,Arrays.copyOf(tmp.al,tmp.al.length)));
						}
					}
				}
				if(tmp.x != 0) {
					if(tmp.al[tmp.x-1]) {
						    queue.add(new Data(tmp.y,tmp.x-1,tmp.c                      ,tmp.o-1                                     ,Arrays.copyOf(tmp.al,tmp.al.length)));
					}
					else {
						if(map[tmp.y][tmp.x-1] > 0 && tmp.o != 1) {
							queue.add(new Data(tmp.y,tmp.x-1,tmp.c						,Math.min(m, tmp.o - 1 + map[tmp.y][tmp.x-1]),Arrays.copyOf(tmp.al,tmp.al.length)));
						}
						else if(map[tmp.y][tmp.x-1] < 0){
							queue.add(new Data(tmp.y,tmp.x-1,tmp.c - map[tmp.y][tmp.x-1],tmp.o-1									 ,Arrays.copyOf(tmp.al,tmp.al.length)));
						}
					}
				}
				if(tmp.y != h-1) {
					if(map[tmp.y+1][tmp.x] > 0 && tmp.o != 1) {
							queue.add(new Data(tmp.y+1,tmp.x,tmp.c						,Math.min(m, tmp.o - 1 + map[tmp.y+1][tmp.x]),new boolean[w]));
					}
					else if(map[tmp.y+1][tmp.x] < 0){
							queue.add(new Data(tmp.y+1,tmp.x,tmp.c - map[tmp.y+1][tmp.x],tmp.o-1									 ,new boolean[w]));
					}
				}
				
			}
			
			if(ans == -1 || ans > f) {
				System.out.println("NA");
			}
			else {
				System.out.println(ans);
			}
		}
		
		
		
		
		
		
		
		
	}
	static class Data implements Comparable<Data> {
		int y;
		int x;
		int c;
		int o;
		boolean[] al;
		
		Data(int y, int x, int c, int o, boolean[] al) {
			this.y = y;
			this.x = x;
			this.c = c;
			this.o = o;
			this.al = al;
		}

		@Override
		public int compareTo(Data o) {
			return this.c - o.c;
		}
		
	}
	
	
	
}