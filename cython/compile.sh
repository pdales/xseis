#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$../pyinclude/"
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:../include/

rm *.so
python3 setup.py build_ext --inplace
cp *.so ../pyinclude/xseis/.
