# Build libcxl_shm.so for the StrataCache CXL backend.
#
# Usage:
#   cd stratacache/csrc/cxl && make
#   # Then either:
#   #   - leave libcxl_shm.so in this directory (binding.py searches here)
#   #   - export STRATACACHE_CXL_LIB=$(pwd)/libcxl_shm.so
#   #   - copy libcxl_shm.so somewhere on LD_LIBRARY_PATH

CC      ?= gcc
CFLAGS  ?= -O2 -fPIC -std=gnu11 -Wall -Wextra -mclflushopt -mclwb
LDFLAGS ?= -shared
LIBS    ?= -lpthread

SRC := cxl_shm.c mem_hash.c
OBJ := $(SRC:.c=.o)
LIB := libcxl_shm.so

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(LIB)
