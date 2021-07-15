include_directories(${PROJECT_SOURCE_DIR}/lib)

AUX_SOURCE_DIRECTORY(./ DIR)

#ADD_EXECUTABLE(main ${DIR})


ADD_EXECUTABLE(Layertest Layertest.cpp)
ADD_EXECUTABLE(main main.cpp)
target_link_libraries(Layertest Mylibs)
TARGET_LINK_LIBRARIES(main Mylibs)