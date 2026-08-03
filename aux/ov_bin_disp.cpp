#include <stdio.h>
#include <errno.h>
#include <string.h>

int main (int argc, char **argv)
{
	FILE *fl;
	float fnum;

	if (argc != 2) {
		printf ("usage: %s, <bin file name>\n", argv[0]);
		return 1;
	}
	if (!(fl = fopen (argv[1], "r"))) {
		perror (strerror(errno));
		return 1;
	}

	printf ("bin file data:\n");
	while (!feof(fl)) {
		fread (&fnum, sizeof(fnum), 1, fl);
		printf ("%f\n",fnum); 
	}
	return 0;
}
