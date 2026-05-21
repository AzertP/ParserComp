{ Free PascalのHello, world.(日本語版) }
 
program Hello;
 
{$MODE OBJFPC}
{$CODEPAGE UTF8}
{$LONGSTRINGS ON}
 
uses
  SysUtils,
  Classes;

procedure MainFunc;
	procedure GetInput(var Length : Integer; var Num : Integer);
		var
			str : string;
			strlist : TStringList;
		begin
			Readln(str);
			strlist := TStringList.Create;
			try
				strlist.Delimiter := ' ';
				strlist.DelimitedText := str;
				Length := StrToInt(strlist.Strings[0]);
				Num := StrToInt(strlist.Strings[1]);
			finally
				FreeAndNil(strlist);
				end;
		end;
	var
		trainLength : Integer;
		trainNum : Integer; 
	begin
		GetInput(trainLength, trainNum);
		WriteLn(trainLength - trainNum + 1);
	end;
	
begin
	MainFunc;
end.
