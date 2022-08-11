#C = g++

# flags for toulbar2 
TOULBARFLAGS = -DLINUX -DLONGLONG_COST -DWIDE_STRING -DLONGDOUBLE_PROB
TOULBARLIBS  = -Ltoulbar2/lib/x86_64 -ltb2

# path to ILOG libraries (CP Optimizer)
ifeq ($(findstring aiopt, $(HOSTNAME)), aiopt)
	ILOGBASE  = /home/jiang631/data/cplex
else
	ILOGBASE  = /home/jiang631/data/cplex
endif
ILOGLIBFORMAT = x86-64_linux/static_pic
ILOGFLAGS = -DILOGLUE -DIL_STD -I$(ILOGBASE)/cpoptimizer/include/ -I$(ILOGBASE)/concert/include/ 
ILOGLIBS  = -L$(ILOGBASE)/cpoptimizer/lib/$(ILOGLIBFORMAT) -L$(ILOGBASE)/concert/lib/$(ILOGLIBFORMAT) -L$(ILOGBASE)/cplex/lib/$(ILOGLIBFORMAT) -lcp -lcplex -lconcert -lm -lpthread

CFLAGS = -std=c++11 -g -Wall $(ILOGFLAGS) $(TOULBARFLAGS) -DNARYCHAR -DWCSPFORMATONLY

#OFLAGS = -O3 -DNDEBUG
# -O1 -DNDEBUG
# -O2 -DNDEBUG

PFLAGS = 
# -pg -g3 # use these options to catch numerical overflows

BUILD = ar rs

default: clean paws 

%.o : %.cpp %.hpp
	$(CC) $(OFLAGS) $(CFLAGS) $(LIBFLAGS) -c -o $@  $< $(PFLAGS)

%.o : %.cpp %.h
	$(CC) $(OFLAGS) $(CFLAGS) $(LIBFLAGS) -c -o $@  $< $(PFLAGS)

%.o : %.cpp
	$(CC) $(OFLAGS) $(CFLAGS) $(LIBFLAGS) -c -o $@  $< $(PFLAGS)

libparity.a: parity.o xors.o xors_cp.o
	$(BUILD) $@ parity.o xors.o xors_cp.o

paws: paws.cpp paws.h libparity.a
	$(CC) $(LFLAGS) -DNDEBUG $(CFLAGS) -o $@ $< -L. -lparity -lgmp $(TOULBARLIBS) $(ILOGLIBS)

clean:
	-rm -f *.o libparity.a paws 

