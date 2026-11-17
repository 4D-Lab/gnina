#from https://www.mattkeeter.com/blog/2018-01-06-versioning/
#
# Run as a script (cmake -P), so the working directory is the build tree. With
# an out-of-tree build that is outside the repository, which means git reports
# nothing even for a proper checkout -- hence GNINA_SOURCE_DIR, passed in by the
# caller. GNINA_VERSION covers the other case: a tree with no .git at all (a
# release archive, or a copy synced without history), which would otherwise
# identify itself as "gnina N/A N/A:N/A".
if(NOT DEFINED GNINA_SOURCE_DIR)
    set(GNINA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

execute_process(COMMAND git log --pretty=format:'%h' -n 1
                WORKING_DIRECTORY "${GNINA_SOURCE_DIR}"
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "N/A")
    set(GIT_DIFF "")
    set(GIT_TAG "N/A")
    set(GIT_BRANCH "N/A")
else()
    execute_process(
        COMMAND bash -c "git diff --quiet --exit-code || echo +"
        WORKING_DIRECTORY "${GNINA_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_DIFF)
    execute_process(
        COMMAND git describe --exact-match --tags
        WORKING_DIRECTORY "${GNINA_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_TAG ERROR_QUIET)
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${GNINA_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_BRANCH)

    string(STRIP "${GIT_REV}" GIT_REV)
    string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
    string(STRIP "${GIT_DIFF}" GIT_DIFF)
    string(STRIP "${GIT_TAG}" GIT_TAG)
    string(STRIP "${GIT_BRANCH}" GIT_BRANCH)
endif()

# An explicitly supplied version stands in for the tag when there is no history.
if("${GIT_REV}" STREQUAL "N/A" AND NOT "${GNINA_VERSION}" STREQUAL "")
    set(GIT_TAG "${GNINA_VERSION}")
endif()

set(VERSION "const char* GIT_REV=\"${GIT_REV}${GIT_DIFF}\";
const char* GIT_TAG=\"${GIT_TAG}\";
const char* GIT_BRANCH=\"${GIT_BRANCH}\";")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp")
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp VERSION_)
else()
    set(VERSION_ "")
endif()

if (NOT "${VERSION}" STREQUAL "${VERSION_}")
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp "${VERSION}")
endif()
