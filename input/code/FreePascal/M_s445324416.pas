var
	M,L,R,ans,ll,rr:int64;
    k,i,x,y,xl,yl,limx,limy,xx,yy,id,ie:Longint;
    dp:array[1..60,1..2,1..2]of int64;
function power(a,b:int64):int64;
var r:int64;
begin
	if b=0 then power:=1
    else begin
    	r:=power(a*a mod M,b div 2);
        if b mod 2=1 then r:=r*a mod M;
        power:=r;
    end;
end;
begin
	read(L,R);
    M:=1000000007;
    ll:=1;
    rr:=2;
    for k:=1 to 60 do begin
    	if(rr<=L)or(R<ll)then begin
        	ll:=ll*2;
            rr:=rr*2;
        	continue;
        end;
        for i:=1 to k do begin
        	dp[i,1,1]:=0;
        	dp[i,1,2]:=0;
        	dp[i,2,1]:=0;
        	dp[i,2,2]:=0;
        end;
        if L<=ll then if rr-1<=R then begin
        	ans:=(ans+power(3,k-1))mod M;
            ll:=ll*2;
            rr:=rr*2;
            continue;
        end else dp[1,2,1]:=1 else if rr-1<=R then dp[1,1,2]:=1 else dp[1,1,1]:=1;
        for i:=1 to k-1 do begin
        	x:=(L>>(k-1-i))mod 2;
            y:=(R>>(k-1-i))mod 2;
            for xl:=1 to 2 do begin
            	if xl=1 then limx:=x else limx:=0;
                for yl:=1 to 2 do begin
                	if dp[i,xl,yl]=0 then continue;
                	if yl=1 then limy:=y else limy:=1;
                    if(limx=1)and(limy=1)then begin
                    	dp[i+1,xl,yl]:=(dp[i+1,xl,yl]+dp[i,xl,yl])mod M;
                    end else if(limx=0)and(limy=1)then begin
                    	dp[i+1,xl,2]:=(dp[i+1,xl,2]+dp[i,xl,yl])mod M;
                        dp[i+1,xl,yl]:=(dp[i+1,xl,yl]+dp[i,xl,yl])mod M;
                        dp[i+1,2,yl]:=(dp[i+1,2,yl]+dp[i,xl,yl])mod M;
                    end else if(limx=0)and(limy=0)then begin
                    	dp[i+1,xl,yl]:=(dp[i+1,xl,yl]+dp[i,xl,yl])mod M;
                    end;
                end;
            end;
        end;
        ll:=ll*2;
        rr:=rr*2;
        ans:=(ans+dp[k,1,1]+dp[k,1,2]+dp[k,2,1]+dp[k,2,2])mod M;
    end;
    writeln(ans);
end.