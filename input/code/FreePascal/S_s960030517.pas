uses
	SysUtils,
	Classes;

procedure MainProc;
	var
		text : string;
		strlist : TStringList;
	begin
	ReadLn(text);
	strlist := TStringList.Create;
	try
		strlist.Delimiter := ' ';
		strlist.DelimitedText := text;
		strlist.Sort;
		if strlist.Count = 3 then
			WriteLn(StrToInt(strlist[0]) + StrToInt(strlist[1]) + (StrToInt(strlist[2]) * 10));
	finally
			FreeAndNil(strlist);
			end;
	end;

begin
MainProc;
end.