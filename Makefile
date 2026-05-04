CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic -fopenmp
LDFLAGS ?= -fopenmp

TARGET := prefix_scan
SRC := src/main.cpp src/scan.cpp
HDR := src/scan.h src/scan.hpp

.PHONY: all clean run test benchmark

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --n 1048579 --threads 4 --repeats 3 --mode both --scan-type exclusive

test: $(TARGET)
	./$(TARGET) --test

benchmark: $(TARGET)
	./scripts/run_benchmarks.sh

clean:
	rm -f $(TARGET) results.csv
