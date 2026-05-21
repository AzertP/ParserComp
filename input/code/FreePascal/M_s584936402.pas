var
   n: integer;
   a: integer;
   b: integer;
   h: array of integer;
   i: integer;
   l: integer;
   r: integer;
   m: integer;
   diff: int64;
   cnt: int64;
begin
   readln(n, a, b);
   setlength(h, n);
   for i := 0 to n - 1 do begin
      readln(h[i]);
   end;
   l := 0;
   r := 1000000000;
   while r - l > 1 do begin
      cnt := 0;
      m := (r + l) div 2;
      for i := 0 to n - 1 do begin
         diff := h[i] - b * m;
         if diff > 0 then begin
            cnt := cnt + (diff div (a - b));
            if diff mod (a - b) <> 0 then begin
               inc(cnt);
            end;
         end;
      end;
      if cnt > m then begin
         l := m;
      end
      else begin
         r := m;
      end;
   end;
   writeln(r);
end.