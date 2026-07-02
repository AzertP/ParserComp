using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AthleteProgramming2
{
    class Program
    {
        static void Main()
        {
            int N = int.Parse(Console.ReadLine());
            int[] array = Console.ReadLine().Split(' ').Select(int.Parse).ToArray();
            int replacementCount = 0;
            replacementCount = SelectionSort(N, array);
            for(int i = 0; i < N - 1; i++)
            {
                Console.Write("{0} ",array[i]);
            }
            Console.WriteLine(array[N-1]);
            Console.WriteLine(replacementCount);
        }

        static int SelectionSort(int N, int[] array)
        {
            int replacementCount = 0;
            for (int i = 0; i < N - 1; i++)
            {
                int minj = i;
                for (int j = i; j < N; j++)
                {
                    if (array[j] < array[minj])
                    {
                        minj = j;
                    }
                }
                if (minj != i)
                {
                    replacementCount++;
                    Swap(ref array[minj], ref array[i]);
                }
            }
            return replacementCount;
        }

        static void Swap(ref int a,ref int b)
        {
            int tmp = a;
            a = b;
            b = tmp;
        }

    }
}
