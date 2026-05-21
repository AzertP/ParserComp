
#include <stdio.h>
#include <stdlib.h>

int index1Origin(int i) {
    return i-1;
}

int main(int argc, char** argv) {
    int nVertices;
    
    // scan number of already existed events
    scanf("%d", &nVertices); 
    
    int adjacencyMatrix[nVertices][nVertices];
//    int vertices;
    int vertex;
    int vertexReached;
    int nEdges;
    int i,j;
    
    // scan already existed events
    for (i = 1; i <= nVertices; i++) {
        scanf("%d", &vertex); 
//        adjacencyMatrix[index1Origin(vertex)];
        
        for (j = 1; j <= nVertices; j++)
            adjacencyMatrix[index1Origin(vertex)][index1Origin(j)] = 0;
        
        scanf("%d", &nEdges); 
                
        for (j = 1; j <= nEdges; j++) {
            scanf("%d", &vertexReached); 
            adjacencyMatrix[index1Origin(vertex)][index1Origin(vertexReached)] = 1;
        }
    }
    
    // print result
    for (i = 1; i <= nVertices; i++) {
        for (j = 1; j <= nVertices; j++) {
            if (j > 1) printf(" ");
            printf("%d", adjacencyMatrix[index1Origin(i)][index1Origin(j)]);
        }
        printf("\n");
    }

    return (EXIT_SUCCESS);
}


