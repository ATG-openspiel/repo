# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /repo/open_spiel/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /repo/open_spiel/examples

# Include any dependencies generated for this target.
include CMakeFiles/value_iteration_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/value_iteration_example.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/value_iteration_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/value_iteration_example.dir/flags.make

CMakeFiles/value_iteration_example.dir/value_iteration_example.o: CMakeFiles/value_iteration_example.dir/flags.make
CMakeFiles/value_iteration_example.dir/value_iteration_example.o: value_iteration_example.cc
CMakeFiles/value_iteration_example.dir/value_iteration_example.o: CMakeFiles/value_iteration_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/repo/open_spiel/examples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/value_iteration_example.dir/value_iteration_example.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/value_iteration_example.dir/value_iteration_example.o -MF CMakeFiles/value_iteration_example.dir/value_iteration_example.o.d -o CMakeFiles/value_iteration_example.dir/value_iteration_example.o -c /repo/open_spiel/examples/value_iteration_example.cc

CMakeFiles/value_iteration_example.dir/value_iteration_example.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/value_iteration_example.dir/value_iteration_example.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /repo/open_spiel/examples/value_iteration_example.cc > CMakeFiles/value_iteration_example.dir/value_iteration_example.i

CMakeFiles/value_iteration_example.dir/value_iteration_example.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/value_iteration_example.dir/value_iteration_example.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /repo/open_spiel/examples/value_iteration_example.cc -o CMakeFiles/value_iteration_example.dir/value_iteration_example.s

# Object files for target value_iteration_example
value_iteration_example_OBJECTS = \
"CMakeFiles/value_iteration_example.dir/value_iteration_example.o"

# External object files for target value_iteration_example
value_iteration_example_EXTERNAL_OBJECTS =

value_iteration_example: CMakeFiles/value_iteration_example.dir/value_iteration_example.o
value_iteration_example: CMakeFiles/value_iteration_example.dir/build.make
value_iteration_example: CMakeFiles/value_iteration_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/repo/open_spiel/examples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable value_iteration_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/value_iteration_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/value_iteration_example.dir/build: value_iteration_example
.PHONY : CMakeFiles/value_iteration_example.dir/build

CMakeFiles/value_iteration_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/value_iteration_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/value_iteration_example.dir/clean

CMakeFiles/value_iteration_example.dir/depend:
	cd /repo/open_spiel/examples && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples/CMakeFiles/value_iteration_example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/value_iteration_example.dir/depend

