using System;
using System.Collections.Generic;

public class RoundRobinScheduling
{
  static void Main(string[] args)
  {
    string[] pName;
    int[] pTime;
    
    string[] input = Console.ReadLine().Split(' ');
    int n = int.Parse(input[0]);
    int q = int.Parse(input[1]);

    pName = new string[n];
    pTime = new int[n];
    
    for( int i = 0; i < n; i++ )
      {
        input = Console.ReadLine().Split(' ');
        pName[i] = input[0];
        pTime[i] = int.Parse(input[1]);
      }

    RoundRobinScheduling rrs = new RoundRobinScheduling();

    rrs.solve(q, pName, pTime);
  }

  public void solve( int q, string[] name, int[] time )
  {
    Queue<string> nq = new Queue<string>();
    Queue<int> tq = new Queue<int>();

    int ft = 0;
    int ct;
    string cn;
    
    for(int i = 0; i < name.Length; i++)
      {
        nq.Enqueue(name[i]);
        tq.Enqueue(time[i]);
        
      }

    while(nq.Count > 0)
      {
        cn = nq.Dequeue();
        ct = tq.Dequeue();
        if(ct > q)
          {
            nq.Enqueue(cn);
            tq.Enqueue(ct-q);
            ft += q;
          }
        else
          {
            ft += ct;
            Console.WriteLine("{0} {1}", cn, ft);      
          }
      }
  }
}
