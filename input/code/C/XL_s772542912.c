#define NDEBUG

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <assert.h>
// for DEBUG {{{
#ifdef NDEBUG
#define DEBUG_PRINT(fmt, ...) ((void)0)
#else
#define DEBUG_PRINT(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#endif
#define ASSERT_RANGE(min,x,max) assert(((x)>=(min)) && ((x)<=(max)))
// }}}

typedef long long ll;

#define GET_D(x) fscanf(stdin,"%d",&(x)) > 0
#define GET_LLD(x) fscanf(stdin,"%lld",&(x)) > 0
#define GET_S(x) fscanf(stdin,"%s",(x)) > 0

#define PUT_D(x) fprintf(stdout,"%d\n",(x))
#define PUT_Ds(x) fprintf(stdout,"%d ",(x))
#define PUT_LLD(x) fprintf(stdout,"%lld\n",(x))
#define PUT_LLDs(x) fprintf(stdout,"%lld ",(x))
#define PUT_LN fprintf(stdout,"\n")

#define FOR(i,min,max) for ((i) = (min); (i) < (max); (i)++)

void merge_int(int *val, int lower, int sep, int upper, int *tmp_val) {// {{{
	int i_lower = lower;
	int i_upper = sep;
	int i_b = 0;
	while ((i_lower < sep) && (i_upper < upper)) {
		if (val[i_lower] < val[i_upper]) {
			tmp_val[i_b++] = val[i_lower++];
		} else {
			tmp_val[i_b++] = val[i_upper++];
		}
	}
	if (i_lower >= sep) {
		while (i_upper < upper) {
			tmp_val[i_b++] = val[i_upper++];
		}
	} else {
		while (i_lower < sep) {
			tmp_val[i_b++] = val[i_lower++];
		}
	}
	for (i_lower = 0; i_lower < i_b; i_lower++) {
		val[i_lower+lower] = tmp_val[i_lower];
	}
}// }}}
void merge_sort_int(int *val, int lower, int upper, int *tmp_val) {// {{{
	int sep;
	if ((lower == upper) || (lower + 1 == upper)) {
		return;
	}
	sep = (lower + upper) / 2;
	merge_sort_int(val,lower,sep,tmp_val);
	merge_sort_int(val,sep,upper,tmp_val);
	merge_int(val,lower,sep,upper,tmp_val);
	return;
}// }}}

void merge_int_int(int *val, int *index, int lower, int sep, int upper, int *tmp_val, int *tmp_index) {// {{{
	int i_lower = lower;
	int i_upper = sep;
	int i_b = 0;
	while ((i_lower < sep) && (i_upper < upper)) {
		if (val[i_lower] < val[i_upper]) {
			tmp_val[i_b] = val[i_lower];
			tmp_index[i_b] = index[i_lower];
			i_b++;
			i_lower++;
		} else {
			tmp_val[i_b] = val[i_upper];
			tmp_index[i_b] = index[i_upper];
			i_b++;
			i_upper++;
		}
	}
	if (i_lower >= sep) {
		while (i_upper < upper) {
			tmp_val[i_b] = val[i_upper];
			tmp_index[i_b] = index[i_upper];
			i_b++;
			i_upper++;
		}
	} else {
		while (i_lower < sep) {
			tmp_val[i_b] = val[i_lower];
			tmp_index[i_b] = index[i_lower];
			i_b++;
			i_lower++;
		}
	}
	for (i_lower = 0; i_lower < i_b; i_lower++) {
		val[i_lower+lower] = tmp_val[i_lower];
		index[i_lower+lower] = tmp_index[i_lower];
	}
}// }}}
void merge_sort_int_int(int *val, int *index, int lower, int upper, int *tmp_val, int *tmp_index) {// {{{
	int sep;
	if ((lower == upper) || (lower + 1 == upper)) {
		return;
	}
	sep = (lower + upper) / 2;
	merge_sort_int_int(val,index,lower,sep,tmp_val,tmp_index);
	merge_sort_int_int(val,index,sep,upper,tmp_val,tmp_index);
	merge_int_int(val,index,lower,sep,upper,tmp_val,tmp_index);
	return;
}// }}}

void merge_ll(ll *val, ll lower, ll sep, ll upper, ll *tmp_val) {// {{{
	ll i_lower = lower;
	ll i_upper = sep;
	ll i_b = 0;
	while ((i_lower < sep) && (i_upper < upper)) {
		if (val[i_lower] < val[i_upper]) {
			tmp_val[i_b++] = val[i_lower++];
		} else {
			tmp_val[i_b++] = val[i_upper++];
		}
	}
	if (i_lower >= sep) {
		while (i_upper < upper) {
			tmp_val[i_b++] = val[i_upper++];
		}
	} else {
		while (i_lower < sep) {
			tmp_val[i_b++] = val[i_lower++];
		}
	}
	for (i_lower = 0; i_lower < i_b; i_lower++) {
		val[i_lower+lower] = tmp_val[i_lower];
	}
}// }}}
void merge_sort_ll(ll *val, ll lower, ll upper, ll *tmp_val) {// {{{
	ll sep;
	if ((lower == upper) || (lower + 1 == upper)) {
		return;
	}
	sep = (lower + upper) / 2;
	merge_sort_ll(val,lower,sep,tmp_val);
	merge_sort_ll(val,sep,upper,tmp_val);
	merge_ll(val,lower,sep,upper,tmp_val);
	return;
}// }}}

void merge_ll_ll(ll *val, ll *index, ll lower, ll sep, ll upper, ll *tmp_val, ll *tmp_index) {// {{{
	ll i_lower = lower;
	ll i_upper = sep;
	ll i_b = 0;
	while ((i_lower < sep) && (i_upper < upper)) {
		if (val[i_lower] < val[i_upper]) {
			tmp_val[i_b] = val[i_lower];
			tmp_index[i_b] = index[i_lower];
			i_b++;
			i_lower++;
		} else {
			tmp_val[i_b] = val[i_upper];
			tmp_index[i_b] = index[i_upper];
			i_b++;
			i_upper++;
		}
	}
	if (i_lower >= sep) {
		while (i_upper < upper) {
			tmp_val[i_b] = val[i_upper];
			tmp_index[i_b] = index[i_upper];
			i_b++;
			i_upper++;
		}
	} else {
		while (i_lower < sep) {
			tmp_val[i_b] = val[i_lower];
			tmp_index[i_b] = index[i_lower];
			i_b++;
			i_lower++;
		}
	}
	for (i_lower = 0; i_lower < i_b; i_lower++) {
		val[i_lower+lower] = tmp_val[i_lower];
		index[i_lower+lower] = tmp_index[i_lower];
	}
}// }}}
void merge_sort_ll_ll(ll *val, ll *index, ll lower, ll upper, ll *tmp_val, ll *tmp_index) {// {{{
	ll sep;
	if ((lower == upper) || (lower + 1 == upper)) {
		return;
	}
	sep = (lower + upper) / 2;
	merge_sort_ll_ll(val,index,lower,sep,tmp_val,tmp_index);
	merge_sort_ll_ll(val,index,sep,upper,tmp_val,tmp_index);
	merge_ll_ll(val,index,lower,sep,upper,tmp_val,tmp_index);
	return;
}// }}}


#define N_MIN 1LL
#define N_MAX 300000LL
#define N_DIV 1000000007LL


int main(int argc, char **argv) {
	ll n;
	ll a[N_MAX];

	GET_LLD(n);
	{
		ll i;
		FOR(i,0,n) {
			GET_LLD(a[i]);
			DEBUG_PRINT("%lld\n",a[i]);
		}
	}
	{
		ll sum = 0;
		ll j;
		FOR(j,0,63) {
			ll n_zero = 0;
			ll n_one = 0;
			ll bitselect = (1LL<<j);
			ll ret;
			ll i;
			FOR(i,0,n) {
				ll tmp = a[i] & bitselect;
				if(tmp != 0) {
					n_one++;
				} else {
					n_zero++;
				}
			}
			DEBUG_PRINT("%2lld %lld %lld\n",j,n_zero,n_one);
			ret = (n_zero * n_one) % N_DIV;
			ret *= (bitselect % N_DIV);
			ret %= N_DIV;
			sum += ret;
			sum %= N_DIV;
		}
		PUT_LLD(sum);
	}
	return 0;
}

