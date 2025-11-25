# Compiler settings
CXX = g++
CXXFLAGS = -I. -std=c++17 -Ofast -pthread -fopenmp -D__AVX__ -mavx2 -pg

# Default target
all:
	@echo "Please specify a target:"
	@echo "  make hnsw   - Build HNSW version"
	@echo "  make acorn  - Build ACORN version"
	@echo "  make clean  - Remove build artifacts"

# HNSW version
hnsw:
	$(CXX) $(CXXFLAGS) hnsw_Wolverine_test.cpp -o hnsw_test

# ACORN version  
acorn:
	$(CXX) $(CXXFLAGS) acorn_Wolverine_test.cpp -o acorn_test

# Clean
clean:
	rm -f hnsw_test acorn_test

.PHONY: all hnsw acorn clean