WARNING: Untested code provided more as an example than a polished product. You may very well want to modify `bindings.cu` to use the cudaKDTree library in a different way.

From the root directory of the repository, run these commands:

cmake -S external -B external/build
cmake --build external/build
cp external/build/libjaxcukd*.so jaxkd/

And you're good to go!