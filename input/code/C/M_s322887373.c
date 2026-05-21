#include <stdio.h>
#include <ctype.h>

/** Application main entry point. */
int
main (
  int     argc,
  char  * argv[ ]
  )
{
  char *it;

  for ( ; ; )
  {
    char s[ 128 ], *p;
    char m;

    scanf ( " %s %c", s, &m );
    if ( m == 'X' ) break ;

    p = s;
    for ( it = s; *it != '\0'; ++it )
    {
      if ( *it == '_' )
      {
        it[ 1 ] = toupper ( it[ 1 ] );
        continue ;
      }
      *( p++ ) = *it;
    }
    *p = '\0';

    s[ 0 ] = tolower ( s[ 0 ] );
    switch ( m )
    {
      case 'U':
        s[ 0 ] = toupper ( s[ 0 ] );
        puts ( s );
        break ;
      case 'L':
        puts ( s );
        break ;
      case 'D':
        for ( it = s; *it != '\0'; ++it )
        {
          if ( isupper ( *it ) )
          {
            printf ( "_%c", tolower ( *it ) );
          }
          else
          {
            putchar ( *it );
          }
        }
        puts ( "" );
    }
  }

  return ( 0 );
}