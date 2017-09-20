objects = float_bitmap.o base.o font.o image_reconstruction.o stopwatch.o std_lucky.o snt_lucky.o metric.o image_utils.o nrfunc.o  im_construct.o parser.o preprocess.o
CC = g++
defs = -fno-implicit-templates
includes = -I/usr/local/Cellar/pgplot/
profile_flags = 
optim_flags =  -O3 
CPPFLAGS = -g $(defs) $(includes) $(profile_flags) $(optim_flags)
libraries = -lm -lcfitsio -L/usr/local/Cellar/pgplot -lcpgplot -lpgplot -L/usr/X11R6/lib -lX11
outfile = image_reconstruction

lucky: $(objects)
	gfortran $(profile_flags) $(optim_flags) -g -L/usr/local/lib $(includes) -o $(outfile) $(objects) -lstdc++  $(libraries) 

%.o: %.cpp 
	$(CC) $(CPPFLAGS) -c $<

all :
	$(outfile)

clean :
	rm -f $(objects) 
	rm -f $(outfile)

