# g++ Process.cpp -std=c++11 -o main
# g++ main.cpp -std=c++11 -pthread -o main
# ./main
rm *.so
python setup.py build_ext --inplace
cp *.so ../pyinclude/.


