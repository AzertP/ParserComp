import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

  public static void main(String[] args) {
    
    new Main().run();
  }

  public void run() {

    try(BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {

      final int n = Integer.parseInt(br.readLine());
      String op;
      String[] words;
      Tree tree = new Tree();
      int t;

      for(int i=0,k=0; i<n; i++,k=0){
        words = br.readLine().split(" ");
        op  = words[k++];

        switch( op ) {
          case "insert" :
            tree.insert(Integer.parseInt(words[k]));
            break;
          case "find" :
            t = Integer.parseInt(words[k]);
            System.out.println( tree.find(t) ? "yes" : "no" );
            break;
          case "delete" :
            tree.delete(Integer.parseInt(words[k]));
            break;
          default :
            tree.dispRoundPath();
        }
      }

    }
    catch(IOException e) {
      System.out.println("IOException!");
    }
  }
}

class Tree {

  static final BTreeNode NIL = new BTreeNode(Integer.MIN_VALUE);
  BTreeNode root;
  StringBuilder buf;

  public Tree() {
    root = NIL;
    buf = new StringBuilder();
  }

  public void insert(int val) {
    BTreeNode y = NIL;
    BTreeNode x = root;
    BTreeNode z = new BTreeNode(val);

    while( !x.equals(NIL) ) {
      y = x;
      x = z.key < x.key ? x.left : x.right;
    }
    z.parent = y;

    if( y.equals(NIL) )
      root = z;
    else if( z.key < y.key )
      y.left = z;
    else
      y.right = z;
  }
  boolean find(int target) {
    BTreeNode x = root;

    while( !x.equals(NIL) ) {
      if( target == x.key )
        return true;
      else
        x = target < x.key ? x.left : x.right;
    }

    return false;
  }
  void delete(int target) {
    BTreeNode x = root;

    while( !x.equals(NIL) ) {
      if( target == x.key )
        break;
      else
        x = target < x.key ? x.left : x.right;
    }

    deleteNode(x);
  }
  void deleteNode(BTreeNode x) {

    if( x.left.equals(NIL) && x.right.equals(NIL) ) {
      int target = x.key;
      x = x.parent;
      if(x.left.key == target) x.left = NIL;
      else x.right = NIL;
    }
    else if( !x.left.equals(NIL) && !x.right.equals(NIL) ) {
      BTreeNode t = getSuccessor(x);
      x.key = t.key;
      deleteNode(t);
    }
    else {
      int target = x.key;
      BTreeNode t;
      if( !x.left.equals(NIL) ) t = x.left;
      else t = x.right;
      x = x.parent;

      if(x.left.key == target) x.left = t;
      else x.right = t;

      t.parent = x;
    }
  }
  BTreeNode getSuccessor(BTreeNode x) {
    if(!x.right.equals(NIL)) return getMinimum(x.right);

    BTreeNode p = x.parent;
    while(!p.equals(NIL) && x.equals(p.right)) {
      x = p;
      p = p.parent;
    }

    return p;
  }
  BTreeNode getMinimum(BTreeNode x) {
    while(!x.left.equals(NIL)) {
      x = x.left;
    }

    return x;
  }  
  void preParse(BTreeNode u) {
    if( !u.equals(NIL) ) {
      buf.append(" ").append(u.key);
      preParse(u.left);
      preParse(u.right);
    }
  }
  void inParse(BTreeNode u) {
    if( !u.equals(NIL) ) {
      inParse(u.left);
      buf.append(" ").append(u.key);
      inParse(u.right);
    }
  }

  void dispRoundPath() {

    inParse(root);
    buf.append("\n");
    preParse(root);
    buf.append("\n");

    System.out.print(buf);
    buf.setLength(0);
  }
}

class BTreeNode {
  int key;
  BTreeNode parent,left,right;

  public BTreeNode(int key) {
    this.key = key;
    this.parent =
    this.left =
    this.right = Tree.NIL;
  }

  @Override
  public boolean equals(Object o) {
    if(o == this) return true;
    if(o == null) return false;
    if(!(o instanceof BTreeNode)) return false;

    BTreeNode node = (BTreeNode)o;
    if(!(this.key == node.key)) return false;

    return true;
  }
}
