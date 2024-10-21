# Define the compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -std=c++14
GENCODE_FLAGS = \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_53,code=sm_53 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_62,code=sm_62 \
    -gencode arch=compute_70,code=sm_70
LIBS = -lhdf5 -lhdf5_cpp

# Target executable
TARGET = test

# Source files
SRCS = test.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Default target to build
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LIBS)

# Rule to compile CUDA source files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# Clean up the generated files
clean:
	rm -f $(OBJS) $(TARGET)
