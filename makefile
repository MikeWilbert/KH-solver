# Compiler
CXX = mpic++
CXXFLAGS = -O2 -Wall -std=c++11

# Target
TARGET = mike_phy

# Rules
all: $(TARGET)

$(TARGET): main.o simulation.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cc include.h
	$(CXX) $(CXXFLAGS) -c $<

simulation.o: simulation.cc simulation.h array_ND.h include.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(TARGET) *.o
