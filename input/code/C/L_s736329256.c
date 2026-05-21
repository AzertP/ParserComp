#include <stdio.h>

int main(){
int a,n,k,i,ii,flag;
scanf("%d %d",&n,&k);
int d[9] ;
for(i = 0;i < 9;i++)
	d[i] = 999;
int m = 0;
int t = n / 1000;
	n = n % 1000;
int h = n / 100;
	n = n % 100;
int z = n / 10;
	n = n % 10;
int o = n ;
int gm = m;
int gt = t;
int gh = h;
int gz = z;
int r = 1;

for(i = 0;i < k;i++)
	scanf("%d",&d[i]);

	for(a = 0;a < k;a++){
		for(i = 0;i < k;i++){
			while(r == 1){
				for(ii = 0;ii < k;ii++){
					if(o == d[ii]){
						o += 1;
					if(o == 10){
						o = 0;
						z += 1;
						}
						r = 0;
						break;
						} 
				
				r = 2;
				}
			}
			if(r == 2)
				break;
			if(o == d[i]){
				o += 1;
				r = 1;
			}
		}
}
r = 1;
	if(z >= 1){for(a = 0;a < k;a++){
		for(i = 0;i < k;i++){
			if(z == 10){
				z = 0;
				h += 1;
			}
			while(r == 1){
				for(ii = 0;ii < k;ii++){
					if(z == d[ii]){
						z += 1;
					if(z == 10){
						z = 0;
						h += 1;
						}
						r = 0;
						break;
						} 
				
				r = 2;
				}
			}
			if(r == 2)
				break;
			if(z == d[i]){
				z += 1;
				r = 1;
			}
		}
}
if(z != gz &&(d[0] != 0 && d[1] != 0 && d[2] != 0 && d[3] != 0 && d[4] != 0 && d[5] != 0 && d[6] != 0 && d[7] != 0 && d[8] != 0))
o = 0;
if(z != gz && o != 0 &&(d[0] != 1 && d[1] != 1&& d[2] != 1&& d[3] != 1&& d[4] != 1&& d[5] != 1 & d[6] != 1&& d[7] != 1&& d[8] != 1))
o = 1;
}

r = 1;
	if(h >= 1){for(a = 0;a < k;a ++){
		for(i = 0;i < k;i++){
			if(h == 10){
				h = 0;
				t += 1;
			}
			while(r == 1){
				for(ii = 0;ii < k;ii++){
					if(h == d[ii]){
						h += 1;
					if(h == 10){
						h = 0;
						t += 1;
						}
						r = 0;
						break;
						} 

				r = 2;
				}
			}
			if(r == 2)
				break;
			if(h == d[i]){
				h += 1;
				r = 1;
			}
		}
}
if(h != gh &&(d[0] != 0 && d[1] != 0 && d[2] != 0 && d[3] != 0 && d[4] != 0 && d[5] != 0 && d[6] != 0 && d[7] != 0 && d[8] != 0))
z = 0;
if(h != gh && z != 0 &&(d[0] != 1 && d[1] != 1&& d[2] != 1&& d[3] != 1&& d[4] != 1&& d[5] != 1 & d[6] != 1&& d[7] != 1&& d[8] != 1))
z = 1;
}

r = 1;
	if(t >= 1){for(a = 0;a < k;a++){
		for(i = 0;i < k;i++){
			if(t == 10){
				t = 0;
				m += 1;
			}
			while(r == 1){
				for(ii = 0;ii < k;ii++){
					if(t == d[ii]){
						t += 1;
					if(t == 10){
						t = 0;
						m += 1;
						}
						r = 0;
						break;
						} 

				r = 2;
				}
			}
			if(r == 2)
				break;
			if(t == d[i]){
				t += 1;
				r = 1;
			}
		}
}
if(t != gt &&(d[0] != 0 && d[1] != 0 && d[2] != 0 && d[3] != 0 && d[4] != 0 && d[5] != 0 && d[6] != 0 && d[7] != 0 && d[8] != 0))
h = 0;}

r = 1;
if(m >= 1){
	for(a = 0;a < k;a++){
		for(i = 0;i < k;i++){
			while(r == 1){
				for(ii = 0;ii < k;ii++){
					if(m == d[ii]){
						m += 1;
						r = 0;
						break;
						} 

				r = 2;
				}
			}
			if(r == 2)
				break;
			if(m == d[i]){
				m += 1;
				r = 1;
			}
		}
}
if(m != gm&&(d[0] != 0 && d[1] != 0 && d[2] != 0 && d[3] != 0 && d[4] != 0 && d[5] != 0 && d[6] != 0 && d[7] != 0 && d[8] != 0))
t = 0;}
		printf("%d",m * 10000 + 1000 * t + 100 * h + 10 * z + 1 * o);
		return 0;
}