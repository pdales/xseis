

# g++-7 $fullfile -std=c++17 -Wall -o temp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lfftw3f_threads -lm -fopenmp -lmseed -lstdc++fs && time ./temp

# g++7 -std=c++17 -Wall -Wextra -shared -o libxseis.so -fPIC xs0.cpp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lfftw3f_threads -lm -fopenmp -lmseed -lstdc++fs


# Creates .so file (libxseis.so) from sample source file (xs0.cpp)
g++7 -std=c++17 -Wall -Wextra -shared -o libxseis.so -fPIC iloc.cpp


# # Tests linking of .so file (main.cpp calls function from libxseis) 
# g++7 -std=c++17 -Wall main.cpp -L$PWD -lxseis -o main


# # Creates .so file (libxseis.so) from sample source file (xs0.cpp)
# g++7 -std=c++17 -Wall -Wextra -shared -o libiloc.so -fPIC iloc.cpp
# # Tests linking of .so file (main.cpp calls function from libxseis) 
# g++7 -std=c++17 -Wall main.cpp -L$PWD -lxseis -o main
# ./main

