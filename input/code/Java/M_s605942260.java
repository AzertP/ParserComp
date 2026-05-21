import java.util.*;
public class Main{
	public static void unite(Node x, Node y){
		leader(y).parent = leader(x);
	}
	public static Node leader(Node x){
		if(x.parent == x)
			return x;
		else
			return x.parent = leader(x.parent);
	}
	public static int same(Node x, Node y){
		if(leader(x) == leader(y))
			return 1;
		else
			return 0;
	}
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		int q = sc.nextInt();
		Node[] nodes = new Node[n];
		for(int i = 0 ; i < n ; i++)
			nodes[i] = new Node(i,true);
		for(int i = 0 ; i < q ; i++){
			int com = sc.nextInt();
			int x = sc.nextInt();
			int y = sc.nextInt();
			switch(com){
				case 0: unite(nodes[x],nodes[y]); break;
				case 1: System.out.println(same(nodes[x],nodes[y])); break;
			}
		}
	}
}
class Node{
	int isRoot;
	int data;
	Node parent = this;
	Node(int data, boolean isroot){
		this.data = data;
		if(isroot)
			isRoot = 1;
		else
			isRoot = 0;
	}
}
