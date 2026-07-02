using System;

class myclass
{
    public static void Main()
    {
        int[] l = new int[10001];
        int x;
        for (int i = 0; ; i++){
            x = Int32.Parse(Console.ReadLine());
            if (x == 0){
                x = i;
                break;
            }
            else
                l[i] = x;
        }
        for (int i = 0; i < x; i++)
        {
            Console.WriteLine("Case "+(i+1)+": "+l[i]);
        }
    }
}
