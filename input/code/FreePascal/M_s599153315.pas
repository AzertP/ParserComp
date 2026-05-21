uses math ;
var a:array[1..100000] of int64 ;
      i,n:longint ;
			ans1,ans2,sum1,sum2:int64 ;
begin 
   readln(n) ;
	 for i:=1 to n do read(a[i]) ;
	 if a[1]>0 then 
	 begin 
	   sum1:=a[1];
		 ans2:=a[1]+1 ;
		 sum2:=-1 ;
	 end
	 else if (a[1]=0) then 
	 begin 
	   sum1:=1;
		 ans1:=1;ans2:=1 ;
		 sum2:=-1 ;
	 end
	 else 
	 begin 
	   sum1:=1;
		 ans1:=abs(a[1])+1 ;
		 sum2:=a[1] ;
	 end;
	 for i:=2 to n do 
	 begin 
	   if sum1>0 then
		 begin 
		   if a[i]+sum1>=0 then 
			 begin 
			   ans1:=ans1+a[i]+sum1+1 ;
				 sum1:=-1 ;
			 end
			 else sum1:=sum1+a[i] ;
		 end
		 else 
		 begin 
		   if a[i]+sum1<=0 then 
			 begin 
			   ans1:=ans1+abs(sum1+a[i])+1 ;
				 sum1:=1 ;
			 end
			 else sum1:=sum1+a[i] ;
		 end;
	 end;
	 if sum1=0 then inc(ans1) ;
	 for i:=2 to n do //这段跟上段一模一样
	 begin 
	   if sum2>0 then
		 begin 
		   if a[i]+sum2>=0 then 
			 begin 
			   ans2:=ans2+a[i]+sum2+1 ;
				 sum2:=-1 ;
			 end
			 else sum2:=sum2+a[i] ;
		 end
		 else 
		 begin 
		   if a[i]+sum2<=0 then 
			 begin 
			   ans2:=ans2+abs(sum2+a[i])+1 ;
				 sum2:=1 ;
			 end
			 else sum2:=sum2+a[i] ;
		 end;
	 end;
	 if sum2=0 then inc(ans2) ;
	 writeln(min(ans1,ans2)) ;
end.