
long int heap[HEAP_SIZE + 1];
int heap_size;

void maxHeapify(long int heap[], int i)
{
    int l = i * 2;
    int r = i * 2 + 1;
    int largest;
    long int w;

    if ((l <= heap_size) && (heap[l] > heap[i]))
    {
        largest = l;
    }
    else
    {
        largest = i;
    }
    if ((r <= heap_size) && (heap[r] > heap[largest]))
    {
        largest = r;
    }
    if (largest != i)
    {
        w = heap[i];
        heap[i] = heap[largest];
        heap[largest] = w;
        maxHeapify(heap, largest);
    }

    return;
}

void heapIncreaseKey(long int heap[], int i, long int key)
{
    int w;
    if (key < heap[i])
    {
        return;
    }
    heap[i] = key;
    while (i > 1 && heap[i / 2] < heap[i])
    {
        w = heap[i / 2];
        heap[i / 2] = heap[i];
        heap[i] = w;
        i = i / 2;
    }
    return;
}

void maxHeapInsert(long int heap[], long int key)
{
    heap_size = heap_size + 1;
    heap[heap_size] = -1;
    heapIncreaseKey(heap, heap_size, key);
    return;
}

long int heapExtractMax(long int heap[])
{
    long int max;
    if (heap_size < 1)
    {
        return (-1);
    }
    max = heap[1];
    heap[1] = heap[heap_size];
    heap_size = heap_size - 1;
    maxHeapify(heap, 1);
    return (max);
}

int main()
{
    char cmd[30];
    long int key;

    do
    {
        scanf("%s", cmd);
        if (strcmp("insert", cmd) == 0)
        {
            scanf("%ld", &key);
            maxHeapInsert(heap, key);
        }
        else if (strcmp("extract", cmd) == 0)
        {
            printf("%ld\n", heapExtractMax(heap));
        }
    } while (strcmp("end", cmd) != 0);

    return (0);
}
