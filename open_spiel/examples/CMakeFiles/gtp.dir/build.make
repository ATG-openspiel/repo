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
include CMakeFiles/gtp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gtp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gtp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gtp.dir/flags.make

CMakeFiles/gtp.dir/gtp.o: CMakeFiles/gtp.dir/flags.make
CMakeFiles/gtp.dir/gtp.o: gtp.cc
CMakeFiles/gtp.dir/gtp.o: CMakeFiles/gtp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/repo/open_spiel/examples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gtp.dir/gtp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gtp.dir/gtp.o -MF CMakeFiles/gtp.dir/gtp.o.d -o CMakeFiles/gtp.dir/gtp.o -c /repo/open_spiel/examples/gtp.cc

CMakeFiles/gtp.dir/gtp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gtp.dir/gtp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /repo/open_spiel/examples/gtp.cc > CMakeFiles/gtp.dir/gtp.i

CMakeFiles/gtp.dir/gtp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gtp.dir/gtp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /repo/open_spiel/examples/gtp.cc -o CMakeFiles/gtp.dir/gtp.s

# Object files for target gtp
gtp_OBJECTS = \
"CMakeFiles/gtp.dir/gtp.o"

# External object files for target gtp
gtp_EXTERNAL_OBJECTS =

gtp: CMakeFiles/gtp.dir/gtp.o
gtp: CMakeFiles/gtp.dir/build.make
gtp: CMakeFiles/gtp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/repo/open_spiel/examples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gtp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gtp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gtp.dir/build: gtp
.PHONY : CMakeFiles/gtp.dir/build

CMakeFiles/gtp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gtp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gtp.dir/clean

CMakeFiles/gtp.dir/depend:
	cd /repo/open_spiel/examples && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples/CMakeFiles/gtp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/gtp.dir/depend

