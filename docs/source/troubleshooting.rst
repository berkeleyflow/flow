Troubleshooting
********


Below we list a few issues commonly observed in installing and using
Flow. 

Errors installing SUMO
=================

Crash at make -j$nproc
-----------------
Try running `make` instead of `make -j$nproc`. The -j$nproc is an optimization
that may exceed the memory of very old machines. 

