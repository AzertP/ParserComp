var 
    a,b,i,j,k:longint;
    p,s:string;
    pos1,pos2,pos3,pos4,o1,o2,o3,o4,o5,o:longint;
begin
   
readln(p);
s:=p;
o1:=0;
j:=0;
k:=0;
o2:=0;
if p[1]='-' then begin delete(p,1,1); o1:=o1+1; end;
pos1:=pos('+',p);
if pos1>0 then
begin

   pos2:=pos('-',p);
   
   if pos2>0 then  o2:=o2+1; 
   for i:=1  to pos1-2 do
       j:=j*10+ord(p[i])-ord('0');
   for i:=pos1+2+o2  to length(p) do
       k:=k*10+ord(p[i])-ord('0');
      
   if o1=1 then j:=0-j;
   if o2=1 then k:=0-k;
   writeln(j+k);
    readln;
   readln;
   exit;
   end;
    pos1:=pos('*',p);
if pos1>0 then
begin

   pos2:=pos('-',p);
   
   if pos2>0 then  o2:=o2+1; 
   for i:=1  to pos1-2 do
       j:=j*10+ord(p[i])-ord('0');
   for i:=pos1+2+o2  to length(p) do
       k:=k*10+ord(p[i])-ord('0');
    
   if o1=1 then j:=0-j;
   if o2=1 then k:=0-k;

   writeln(j*k);
   readln;
   readln;
   exit;
   end;
    pos1:=pos('\',p);
if pos1>0 then
begin

   pos2:=pos('-',p);
   
   if pos2>0 then  o2:=o2+1; 
   for i:=1  to pos1-2 do
       j:=j*10+ord(p[i])-ord('0');
   for i:=pos1+2+o2  to length(p) do
       k:=k*10+ord(p[i])-ord('0');
    
   if o1=1 then j:=0-j;
   if o2=1 then k:=0-k;
  if k=0 then begin  writeln('NO');
  readln;
  readln;
  exit;
  end;

   writeln(j div k);
   readln;
   readln;
   
   exit;
   end;
   
   pos1:=pos('-',p);
   delete(p,pos1,1);
if pos1>0 then
begin

   pos2:=pos('-',p);
   
   if pos2>0 then  o2:=o2+1; 
   insert('-',p,pos1);
   for i:=1  to pos1-2 do
       j:=j*10+ord(p[i])-ord('0');
   for i:=pos1+2+o2  to length(p) do
       k:=k*10+ord(p[i])-ord('0');
      
   if o1=1 then j:=0-j;
   if o2=1 then k:=0-k;
  
   writeln(j-k);

    readln;
   readln;
   
   exit;
   end;
  

end.