program ec12;
var 
	ans,sum1,sum2,ans1:int64;
	a:array[0..100000] of int64;
	n,m,i,j:longint;
begin 	
	readln(n);
	ans:=0;
	ans1:=0;
	for i:=1 to n do
	read(a[i]);
	if a[1]>0 then 
	begin 
		sum1:=a[1];
		ans1:=a[1]+1;
		sum2:=-1;
	end
	else
	begin
		if a[1]=0 then 
		begin 
			sum1:=1;
			sum2:=-1;
			ans:=1;
			ans1:=1;
		end
		else
		begin
			ans:=abs(a[1])+1;
			sum1:=1;
			sum2:=a[1];
		end;
	end;
	for i:=2 to n do 
	begin 
		if sum1>0 then 
		begin 
			if sum1+a[i]>=0 then 
			begin 	
				inc(ans,sum1+a[i]+1);
				sum1:=-1;
			end
			else
			sum1:=sum1+a[i];
		end
		else
		begin 	
			if sum1+a[i]<=0 then 
			begin 
				inc(ans,abs(sum1+a[i])+1);
				sum1:=1;
			end
			else
			sum1:=sum1+a[i];
		end;
		if sum2>0 then 
		begin 
			if sum2+a[i]>=0 then 
			begin 	
				inc(ans1,sum2+a[i]+1);
				sum2:=-1;
			end
			else
			sum2:=sum2+a[i];
		end
		else
		begin 	
			if sum2+a[i]<=0 then 
			begin 
				inc(ans1,abs(sum2+a[i])+1);
				sum2:=1;
			end
			else
			sum2:=sum2+a[i];
		end;
	end;
	if sum1=0 then 
	inc(ans);
	if sum2=0 then 
	inc(ans1);
	if ans<ans1 then 
	writeln(ans)
	else
	writeln(ans1);
end. 