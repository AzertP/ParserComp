

int get_int(void) {
  int num;
  char line[BUF_SIZE];
  if(!fgets(line, BUF_SIZE, stdin)) return 0;
  sscanf(line, "%d", &num);
  return num;
}


int main(void) {
    static uint64_t parts[PARTS_MAX*2];
    int ps = get_int();
    uint64_t sum3 = 0;
    static uint64_t cum[PARTS_MAX*2+1];
    uint64_t part_max = 0;
    int i;
    for(i = 0; i < ps; i++) {
        parts[i] = parts[i+ps] = get_int();
        part_max = max(part_max, parts[i]);
    }
    for(i = 1; i <= ps*2; i++) {
        cum[i] = parts[i-1] + cum[i-1];
    }
    // max in theory.
    uint64_t sum = cum[ps];
    sum3 = max(sum / 3, part_max);
    printf("sum3: %llu\n", sum3);

    int start, end;
    uint64_t ans = 0;
    uint64_t s = 0;
    for(start = 0, end = 0; start < ps; start++) {
        // Shactory Method .. set end
        while(end < start+ps) {
            uint64_t ns = s + parts[end];
            if(ns > sum3) break;
            s = ns;
            end++;
        }
        printf("[%d, %d)\n", start, end);
        // [start, end-1], [end, high), [high, start+ps)

        // search [end, start+ps) in [start, start+ps) using binary-search.
        int low = end-1;
        int high = start + ps;
        while(low + 1 < high) {
            int mid = (low + high)/2;
            uint64_t vs = cum[mid] - cum[end]; // [end, mid)
            // lower bound
            if(s <= vs) {
                high = mid;
            } else {
                low = mid;
            }
        }
        // update answer
        // [end, high)
        uint64_t s2 = cum[high] - cum[end];
        uint64_t rem = sum - s - s2;
        uint64_t res = min(s, rem);
        printf("[%d, %d), [%d, %d+1) -> %llu ;;", start, end, end, high, s);
        printf("%llu %llu %llu\n", s, s2, rem);
        ans = max(ans, res);

        // update s
        s -= parts[start];
    }
    printf("%llu\n", ans);
    return 0;
}
