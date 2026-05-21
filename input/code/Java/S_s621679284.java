import java.io.*;
class Main
{
    public static void main(String args[])throws IOException
    {
	BufferedReader input=new BufferedReader(new InputStreamReader(System.in));
	String str1;
	int count=0;
	while((str1=input.readLine())!=null)
	    {
		char str2[]=new char[str1.length()];
		for(int i=0;i<str1.length();i++)
		    {
			str2[i]=str1.charAt(str1.length()-(1+i));
		    }
		String str3=String.valueOf(str2);
		
		
		if(str1.equals(str3))count++;
		
	    }
	System.out.println(count);
    }
}