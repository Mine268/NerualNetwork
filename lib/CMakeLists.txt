
aux_source_directory(. DIRS)

add_library(Mylibs STATIC ${DIRS} FileReader.cpp FileReader.h)
