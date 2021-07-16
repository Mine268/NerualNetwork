include_directories(${PROJECT_SOURCE_DIR}/lib)

AUX_SOURCE_DIRECTORY(./ DIR)

#ADD_EXECUTABLE(main ${DIR})

#TARGET_LINK_LIBRARIES(main Mylibs)
ADD_EXECUTABLE(Layertest Layertest.cpp)
ADD_EXECUTABLE(main main.cpp)
target_link_libraries(Layertest Mylibs)