
/*
 * 二叉树 
 * ALDS1_7_B
 * 
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;



public class Main {
	
	
	
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String line = "";
		while ((line = br.readLine()) != null && !line.isEmpty()) {

			int n = Integer.parseInt(line);
			Tree tree = new Tree();

			// ??????????則????
			for (int i = 0; i < n; i++) {
				tree.addNode(br.readLine());
			}
			// ??????????????賊??????
			for (int i = 0; i < n; i++) {
				System.out.println(tree.getNode(i).toString());
			}
		}
		
	}
	
	
	public static class Tree {
		ArrayList<Node> nodes = new ArrayList<Node>();
		
		void addNode(String info)
		{
			int pid, lid, rid;
			pid = Integer.parseInt(info.split(" ")[0]);
			lid = Integer.parseInt(info.split(" ")[1]);
			rid = Integer.parseInt(info.split(" ")[2]);
			
			Node parent, left = null, right = null;
			if((parent=getNode(pid))==null)
			{
				parent = new Node(pid,null);
			}
			
			
			
			if (lid != -1) 
			{
				if ((left = getNode(lid)) == null) 
				{
					left = new Node(lid, parent);
				} 
				else 
				{
					left.parent = parent;
					nodes.remove(left);
				}
			}
			//
			if (rid != -1) 
			{
				if ((right = getNode(rid)) == null)
				{
					right = new Node(rid, parent);
				} 
				else 
				{
					right.parent = parent;
					nodes.remove(right);
				}
			}
			
			parent.left = left;
			parent.right = right;
			
			if (parent.parent == null) 
			{
				nodes.add(parent);
			}
		}
			
			
			
			Node getNode(int id) 
			{
				Node result = null;
				for (Node n : nodes) {
					result = n.find(id);
					if (result != null)
						break;
				}
				return result;
			}
		
	}
	
	
	public static class Node {
		int id;  //节点编号
		Node parent; // 父节点 -1表示没有
		Node left; 
		Node right; //左右子节点 -1表示没有
		
		
		Node(int value, Node parent)
		{
			this.id = value;
			this.parent = parent;
		}
		
		//方法find 返回一个Node
		
		Node find(int id)
		{
			
			//就一直找
			Node result = null;
			if(this.id == id)
			{
				result = this;
			}
			
			if(result == null && left != null)
			{
				result = left.find(id);
			}
			
			if(result == null && right != null)
			{
				result = right.find(id);
			}
			
			return result;
		}
		
		//返回id
		int id()
		{
			return id;
		}
		
		//返回父节点 id
		int parentId()
		{
			if(parent != null)
				return parent.id();
			else
				return -1;
		}
		
		//返回兄弟节点
		int sibling()
		{
			int lid, rid;
			if(parent != null)
			{
				if(parent.left != null)
					lid = parent.left.id();
				else
					lid = -1;
				if(parent.right != null)
					rid = parent.right.id();
				else
					rid = -1;
			
				//这里的lid 和 rid 是父节点的 左子节点和右子节点
				//由于是二叉树
				//lid 和 rid 有一个是id
				//如果自己是lid 及左节点，返回右节点
				//如果自己是rid 及右节点，返回左节点
				return lid != id ? lid : rid;
			}
			else
			{
				//没有父节点，根节点
				return -1;
			}
		}
		
		
		//返回度数，即子节点的个数
		int degree()
		{
			int result = 0;
			if(left != null)
			{
				result++;
			}
			if(right != null)
			{
				result++;
			}
			
			return result;
		}
		
		
		//返回深度
		int depth()
		{
			int result = 0;
			Node p = parent;
			while(p != null)
			{
				result++;
				p = p.parent;
			}
			return result;
		}
		
		
		// 返回高
		int height()
		{
			int lh = 0;
			int rh = 0;
			if(left != null)
			{
				//一路往下，找到没有
				lh = left.height()+1;
			}
			if(right != null )
			{
				rh = right.height()+1;
			}
			return lh>rh ? lh: rh;
		}
		
		//
		String type()
		{
			String result = "";
			if(parent==null)
			{
				return "root";
			}
			if(left!=null||right!=null)
			{
				return "internal node";
			}
			if(left==null&&right==null)
			{
				return "leaf";
			}
			
			return result;
		}
		
		
		public String toString()
		{
			StringBuilder sb = new StringBuilder();
			sb.append("node " + id() + ": ");
			sb.append("parent = " + parentId() + ", ");
			sb.append("sibling = " + sibling() + ", ");
			sb.append("degree = " + degree() + ", ");
			sb.append("depth = " + depth() + ", ");
			sb.append("height = " + height() + ", ");
			sb.append(type());
			return sb.toString();
		}
		
		
	}
	
	
	
	
	
	
}

