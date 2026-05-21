var s:String;i:Longint;f:boolean;
begin
	readln(s);
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='a' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('a');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='b' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('b');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='d' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('d');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='f' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('f');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='i' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('i');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='p' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('p');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='y' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('y');
		exit;
	end;
	f:=false;
	for i:=1 to length(s) do begin
		if s[i]='z' then begin
			f:=true;
			break;
		end;
	end;
	if not f then begin
		writeln('z');
		exit;
	end;
	writeln('None');
end.
