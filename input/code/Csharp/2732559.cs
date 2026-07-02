using System;

namespace g
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                string[] lee = Console.ReadLine().Split();
                if (int.Parse(lee[0]) == 0 && int.Parse(lee[1]) == 0) break;
                Console.WriteLine(Combinaciones(int.Parse(lee[0]), int.Parse(lee[1])));
            }
        }
        public static int cant;
        static int Combinaciones(int n, int x)
        {
            cant = 0;
            Solucion(n, x, 3, 0, 1);
            return cant;
        }
        static void Solucion(int n, int x, int numeros, int suma, int pos)
        {
            if (numeros == 0 && suma == x) { cant++; return; }
            else if (numeros == 0 && suma != x) return;
            for (int i = pos; i <= n; i++)
            {
                if (i + suma <= x)
                {
                    Solucion(n, x, numeros - 1, i + suma, i + 1);
                }
            }
        }
    }
}

