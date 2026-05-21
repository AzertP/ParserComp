import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.Scanner;
import java.util.HashMap;
import java.util.Deque;
import java.util.Objects;
import java.util.Map;
import java.util.LinkedList;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Main
{
    public static void main(String[] args)
    {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        Scanner in = new Scanner(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Task solver = new Task();
        solver.solve(1, in, out);
        out.close();
    }

    static class Task
    {
        public void solve(int testNumber, Scanner in, PrintWriter out)
        {
            int H = in.nextInt();
            int W = in.nextInt();
            char[][] matrix = new char[H][W];
            for (int row = 0; row < H; row++)
            {
                matrix[row] = in.next().toCharArray();
            }

            //SOLVE
            Map<Path, Integer> map = new HashMap<>();
            for (int startRow = 0; startRow < H; startRow++)
            {
                for (int startColumn = 0; startColumn < W; startColumn++)
                {
                    for (int endRow = 0; endRow < H; endRow++)
                    {
                        for (int endColumn = 0; endColumn < W; endColumn++)
                        {
                            if (map.containsKey(getKey(startRow, startColumn, endRow, endColumn)) ||
                                        map.containsKey(getKey(endRow, endColumn, startRow, startColumn))) continue;
                            if ((matrix[startRow][startColumn] != '#' && matrix[endRow][endColumn] != '#') &&
                                        (startRow != endRow || startColumn != endColumn))
                            {

                                int[][] visited = new int[H][W];
                                visited[startRow][startColumn] = 1;
                                find(startRow, startColumn, endRow, endColumn, matrix, visited, map);
                            }
                        }
                    }
                }
            }
            int max = Integer.MIN_VALUE;
            for (Path path : map.keySet())
            {
                int depth = map.get(path);
                max = Math.max(max, depth);
            }
            out.print(max);
        }

        private void find(int startRow, int startColumn, int endRow, int endColumn, char[][] matrix,
                          int[][] visited, Map<Path, Integer> map)
        {

            Deque<Location> q = new LinkedList<>();
            q.addFirst(new Location(startRow + 1, startColumn, 1));
            q.addFirst(new Location(startRow, startColumn + 1, 1));
            q.addFirst(new Location(startRow - 1, startColumn, 1));
            q.addFirst(new Location(startRow, startColumn - 1, 1));
            while (!q.isEmpty())
            {
                Location next = q.pollFirst();
                int currentRow = next.row;
                int currentColumn = next.column;
                int depth = next.depth;

                if (!isValidLocation(currentRow, currentColumn, matrix)) continue;
                if (visited[currentRow][currentColumn] == 1) continue;

                visited[currentRow][currentColumn] = 1;
                if (startRow != endRow || startColumn != endColumn)
                {
                    map.put(getKey(startRow, startColumn, currentRow, currentColumn), depth);
                    map.put(getKey(currentRow, currentColumn, startRow, startColumn), depth);
                }
                if (currentRow == endRow && currentColumn == endColumn) return;

                q.addLast(new Location(currentRow, currentColumn - 1, depth + 1));
                q.addLast(new Location(currentRow, currentColumn + 1, depth + 1));
                q.addLast(new Location(currentRow - 1, currentColumn, depth + 1));
                q.addLast(new Location(currentRow + 1, currentColumn, depth + 1));
            }
        }

        private Path getKey(int startRow, int startColumn, int endRow, int endColumn)
        {
            Path path = new Path();
            path.startRow = startRow;
            path.startColumn = startColumn;
            path.endRow = endRow;
            path.endColumn = endColumn;
            return path;
        }

        private boolean isValidLocation(int currentRow, int currentColumn, char[][] matrix)
        {
            return (currentRow >= 0 && currentRow < matrix.length) &&
                           (currentColumn >= 0 && currentColumn < matrix[0].length) &&
                           matrix[currentRow][currentColumn] == '.';
        }

    }

    static class Location
    {
        int row;
        int column;
        int depth;

        Location(int row, int column, int depth)
        {
            this.row = row;
            this.column = column;
            this.depth = depth;
        }

    }

    static class Path
    {
        int startRow;
        int startColumn;
        int endRow;
        int endColumn;

        public boolean equals(Object o)
        {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Path path = (Path) o;
            return (startRow == path.startRow &&
                            startColumn == path.startColumn &&
                            endRow == path.endRow &&
                            endColumn == path.endColumn);
        }

        public int hashCode()
        {
            return Objects.hash(startRow, startColumn, endRow, endColumn);
        }

    }
}

