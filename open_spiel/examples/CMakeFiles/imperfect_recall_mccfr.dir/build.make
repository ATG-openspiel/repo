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
include CMakeFiles/imperfect_recall_mccfr.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/imperfect_recall_mccfr.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/imperfect_recall_mccfr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imperfect_recall_mccfr.dir/flags.make

CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o: CMakeFiles/imperfect_recall_mccfr.dir/flags.make
CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o: imperfect_recall_mccfr.cc
CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o: CMakeFiles/imperfect_recall_mccfr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/repo/open_spiel/examples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o -MF CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o.d -o CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o -c /repo/open_spiel/examples/imperfect_recall_mccfr.cc

CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /repo/open_spiel/examples/imperfect_recall_mccfr.cc > CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.i

CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /repo/open_spiel/examples/imperfect_recall_mccfr.cc -o CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.s

# Object files for target imperfect_recall_mccfr
imperfect_recall_mccfr_OBJECTS = \
"CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o"

# External object files for target imperfect_recall_mccfr
imperfect_recall_mccfr_EXTERNAL_OBJECTS =

imperfect_recall_mccfr: CMakeFiles/imperfect_recall_mccfr.dir/imperfect_recall_mccfr.o
imperfect_recall_mccfr: CMakeFiles/imperfect_recall_mccfr.dir/build.make
imperfect_recall_mccfr: CMakeFiles/imperfect_recall_mccfr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/repo/open_spiel/examples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable imperfect_recall_mccfr"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imperfect_recall_mccfr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imperfect_recall_mccfr.dir/build: imperfect_recall_mccfr
.PHONY : CMakeFiles/imperfect_recall_mccfr.dir/build

CMakeFiles/imperfect_recall_mccfr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imperfect_recall_mccfr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imperfect_recall_mccfr.dir/clean

CMakeFiles/imperfect_recall_mccfr.dir/depend:
	cd /repo/open_spiel/examples && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples /repo/open_spiel/examples/CMakeFiles/imperfect_recall_mccfr.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/imperfect_recall_mccfr.dir/depend

