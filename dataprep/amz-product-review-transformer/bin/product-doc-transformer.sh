#!/usr/bin/env bash
#
#                      _            _                         
# _ __  _ __ ___   __| |_   _  ___| |_                       
#| '_ \| '__/ _ \ / _` | | | |/ __| __|                      
#| |_) | | | (_) | (_| | |_| | (__| |_                       
#| .__/|_|  \___/ \__,_|\__,_|\___|\__|                      
#|_|                                                         
#     _                                       _              
#  __| | ___   ___ _   _ _ __ ___   ___ _ __ | |_ ___        
# / _` |/ _ \ / __| | | | '_ ` _ \ / _ \ '_ \| __/ __|       
#| (_| | (_) | (__| |_| | | | | | |  __/ | | | |_\__ \       
# \__,_|\___/ \___|\__,_|_| |_| |_|\___|_| |_|\__|___/       
#                                                            
# _                        __                                
#| |_ _ __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __ 
#| __| '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__|
#| |_| | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |   
# \__|_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_| 
#
#
# Wrapper script to execute the Apache Spark Job that transforms
# product reviews documents from their original json into parquet
# format ready for document embedding training.
#
# Usage:
#   product-doc-transformer [options] <argument>
#
# Depends on:
#  list
#  of
#  programs
#  expected
#  in
#  environment
#
# Using: Bash Boilerplate: https://github.com/alphabetum/bash-boilerplate
#
# Copyright (c) 2015 William Melody • hi@williammelody.com

set -o nounset

# Exit immediately if a pipeline returns non-zero.
# Short form: set -e
set -o errexit

# Print a helpful message if a pipeline with non-zero exit code causes the
# script to exit as described above.
trap 'echo "Aborting due to errexit on line $LINENO. Exit code: $?" >&2' ERR

# Allow the above trap be inherited by all functions in the script.
#
# Short form: set -E
set -o errtrace

# Return value of a pipeline is the value of the last (rightmost) command to
# exit with a non-zero status, or zero if all commands in the pipeline exit
# successfully.
set -o pipefail

# Set $IFS to only newline and tab.
#
# http://www.dwheeler.com/essays/filenames-in-shell.html
IFS=$'\n\t'

###############################################################################
# Environment
###############################################################################

# $_ME
#
# Set to the program's basename.
_ME=$(basename "${0}")

###############################################################################
# Debug
###############################################################################

# _debug()
#
# Usage:
#   _debug printf "Debug info. Variable: %s\n" "$0"
#
# A simple function for executing a specified command if the `$_USE_DEBUG`
# variable has been set. The command is expected to print a message and
# should typically be either `echo`, `printf`, or `cat`.
__DEBUG_COUNTER=0
_debug() {
  if [[ "${_USE_DEBUG:-"0"}" -eq 1 ]]
  then
    __DEBUG_COUNTER=$((__DEBUG_COUNTER+1))
    # Prefix debug message with "bug (U+1F41B)"
    printf "🐛  %s " "${__DEBUG_COUNTER}"
    "${@}"
    printf "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――\\n"
  fi
}
# debug()
#
# Usage:
#   debug "Debug info. Variable: $0"
#
# Print the specified message if the `$_USE_DEBUG` variable has been set.
#
# This is a shortcut for the _debug() function that simply echos the message.
debug() {
  _debug echo "${@}"
}

###############################################################################
# Die
###############################################################################

# _die()
#
# Usage:
#   _die printf "Error message. Variable: %s\n" "$0"
#
# A simple function for exiting with an error after executing the specified
# command. The command is expected to print a message and should typically
# be either `echo`, `printf`, or `cat`.
_die() {
  # Prefix die message with "cross mark (U+274C)", often displayed as a red x.
  printf "❌  "
  "${@}" 1>&2
  exit 1
}
# die()
#
# Usage:
#   die "Error message. Variable: $0"
#
# Exit with an error and print the specified message.
#
# This is a shortcut for the _die() function that simply echos the message.
die() {
  _die echo "${@}"
}

###############################################################################
# Help
###############################################################################

# _print_help()
#
# Usage:
#   _print_help
#
# Print the program help information.
_print_help() {
  cat <<HEREDOC

                     _            _                         
 _ __  _ __ ___   __| |_   _  ___| |_                       
| '_ \\| '__/ _ \\ / _\` | | | |/ __| __|                      
| |_) | | | (_) | (_| | |_| | (__| |_                       
| .__/|_|  \___/ \\__,_|\\__,_|\\___|\\__|                      
|_|                                                         
     _                                       _              
  __| | ___   ___ _   _ _ __ ___   ___ _ __ | |_ ___        
 / _\` |/ _ \\ / __| | | | '_ \` _ \\ / _ \\ '_ \\| __/ __|       
| (_| | (_) | (__| |_| | | | | | |  __/ | | | |_\\__ \\       
 \\__,_|\\___/ \\___|\\__,_|_| |_| |_|\\___|_| |_|\\__|___/       
                                                            
 _                        __                                
| |_ _ __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __ 
| __| '__/ _\` | '_ \\/ __| |_ / _ \\| '__| '_ \` _ \\ / _ \\ '__|
| |_| | | (_| | | | \\__ \\  _| (_) | |  | | | | | |  __/ |   
 \\__|_|  \\__,_|_| |_|___/_|  \\___/|_|  |_| |_| |_|\\___|_| 

Wrapper script to execute the Apache Spark Job that transforms
tweets from their original json into parquet format and some
constraints.

Usage:
  ${_ME} [--options]
  ${_ME} -h | --help

Options:
  -h --help               Display this help information.

Spark Related Options:
  -sh --spark-home        Spark home path (Default: environment variable SPARK_HOME)
  -m  --master            Spark master (Default: 'local[*]')
  -dm --driver-memory     Spark driver memory (Default: 16G)
  -em --executor-memory   Spark executor memory (Default: 7G)
  -dc --driver-cores      Spark driver cores (Default: 12)
  -ec --executor-cores    Spark executor cores (Default: 45)
  -ne --num-executors     Spark number of executors (Default: 12)
  -el --event-log         Location to save the spark event logs (Default: /tmp/spark-event)
  -jj --job-jar           Spark Job Jar path (Default: ./product-doc-transformer.jar)

Job Related Options:
  -o  --output            Output directory [Required.]
  -md --metadata          Product metadata file path [Required.]
  -rv --reviews           Product reviews file path [Required.]
  -cm --category-map      Product category mapping file (Default: ./category_mappings.yml)
  -l  --limit             Maximum number of records to process (Default: 10000000)
  -s  --seed              Random seed for sampling             (Default: 12345)
  -ac --array-categories  Set if you want the categories per document as an array instead of a comma separated string.

HEREDOC
}

###############################################################################
# Options
#
# NOTE: The `getops` builtin command only parses short options and BSD `getopt`
# does not support long arguments (GNU `getopt` does), so the most portable
# and clear way to parse options is often to just use a `while` loop.
#
# For a pure bash `getopt` function, try pure-getopt:
#   https://github.com/agriffis/pure-getopt
#
# More info:
#   http://wiki.bash-hackers.org/scripting/posparams
#   http://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html
#   http://stackoverflow.com/a/14203146
#   http://stackoverflow.com/a/7948533
#   https://stackoverflow.com/a/12026302
#   https://stackoverflow.com/a/402410
###############################################################################

# Parse Options ###############################################################

# Initialize program option variables.
_PRINT_HELP=0
_USE_DEBUG=0

# Initialize additional expected option variables.
_SPARK_HOME=${SPARK_HOME:-""}
_DEFAULT_MASTER="local[*]"
_DEFAULT_DRIVER_MEMORY="16G"
_DEFAULT_EXECUTOR_MEMORY="7G"
_DEFAULT_DRIVER_CORES=12
_DEFAULT_EXECUTOR_CORES=45
_DEFAULT_NUM_EXECUTORS=12
_DEFAULT_EVENT_LOG_PATH="/tmp/spark-event"
_DEFAULT_JOB_JAR="./product-doc-transformer.jar"

_METADATA_PATH=""
_REVIEWS_PATH=""
_CATEGORY_MAP_PATH="./category_mappings.yml"
_LIMIT=10000000
_SEED=12345
_ARRAY_CAT=0
_OUTPUT_PATH=""

# _require_argument()
#
# Usage:
#   _require_argument <option> <argument>
#
# If <argument> is blank or another option, print an error message and  exit
# with status 1.
_require_argument() {
  # Set local variables from arguments.
  #
  # NOTE: 'local' is a non-POSIX bash feature and keeps the variable local to
  # the block of code, as defined by curly braces. It's easiest to just think
  # of them as local to a function.
  local _option="${1:-}"
  local _argument="${2:-}"

  if [[ -z "${_argument}" ]] || [[ "${_argument}" =~ ^- ]]
  then
    _die printf "Option requires an argument: %s\\n" "${_option}"
  fi
}

while [[ ${#} -gt 0 ]]
do
  __option="${1:-}"
  __maybe_param="${2:-}"
  case "${__option}" in
    -h|--help)
      _PRINT_HELP=1
      ;;
    --debug)
      _USE_DEBUG=1
      ;;
    -ac|--array-categories)
      _ARRAY_CAT=1
      ;;
    -sh|--spark-home)
      _SPARK_HOME="${__maybe_param}"
      shift
      ;;
    -m|--master)
      _DEFAULT_MASTER="${__maybe_param}"
      shift
      ;;
    -dm|--driver-memory)
      _DEFAULT_DRIVER_MEMORY="${__maybe_param}"
      shift
      ;;
    -em|--executor-memory)
      _DEFAULT_EXECUTOR_MEMORY="${__maybe_param}"
      shift
      ;;
    -dc|--driver-cores)
      _DEFAULT_DRIVER_CORES="${__maybe_param}"
      shift
      ;;
    -ec|--executor-cores)
      _DEFAULT_EXECUTOR_CORES="${__maybe_param}"
      shift
      ;;
    -ne|--num-executor)
      _DEFAULT_NUM_EXECUTORS="${__maybe_param}"
      shift
      ;;
    -el|--event-log)
      _DEFAULT_EVENT_LOG_PATH="${__maybe_param}"
      shift
      ;;
    -jj|--job-jar)
      _DEFAULT_JOB_JAR="${__maybe_param}"
      shift
      ;;
    -o|--output)
      _require_argument "${__option}" "${__maybe_param}"
      _OUTPUT_PATH="${__maybe_param}"
      shift
      ;;
    -md|--metadata)
      _require_argument "${__option}" "${__maybe_param}"
      _METADATA_PATH="${__maybe_param}"
      shift
      ;;
    -rv|--reviews)
      _require_argument "${__option}" "${__maybe_param}"
      _REVIEWS_PATH="${__maybe_param}"
      shift
      ;;
    -cm|--category-map)
      _require_argument "${__option}" "${__maybe_param}"
      _CATEGORY_MAP_PATH="${__maybe_param}"
      shift
      ;;
    -l|--limit)
      _LIMIT="${__maybe_param}"
      shift
      ;;
    -s|--seed)
      _SEED="${__maybe_param}"
      shift
      ;;
    --endopts)
      # Terminate option parsing.
      break
      ;;
    -*)
      _die printf "Unexpected option: %s\\n" "${__option}"
      ;;
  esac
  shift
done

###############################################################################
# Program Functions
###############################################################################

_spark_submit() {

  _debug printf ">> Performing operation...\\n"

  if [ -z "${_OUTPUT_PATH}" ];
  then
    _die printf "Output path is required.\\n"
  fi

  if [ -z "${_SPARK_HOME}" ];
  then
    _die printf "There is no SPARK_HOME environment variable defined nor it was provided as option. This is required for the program to run.\\n"
  fi

  if [ -z "${_METADATA_PATH}" ];
  then
    _die printf "The metadata path is required.\\n"
  fi

  if [ -z "${_REVIEWS_PATH}" ];
  then
    _die printf "The reviews path is required.\\n"
  fi


  strcat=""
  if [ "${_ARRAY_CAT}" -eq "1" ];
  then
    strcat="--strcat"
  fi

  time ${_SPARK_HOME}/bin/spark-submit --class co.edu.icesi.wtsp.amz.product.review.transformer.AmzProductReviewTransformerApp \
  --master ${_DEFAULT_MASTER} \
  --driver-memory ${_DEFAULT_DRIVER_MEMORY} \
  --executor-memory ${_DEFAULT_EXECUTOR_MEMORY} \
  --driver-cores ${_DEFAULT_DRIVER_CORES} \
  --total-executor-cores ${_DEFAULT_EXECUTOR_CORES} \
  --num-executors ${_DEFAULT_NUM_EXECUTORS} \
  --conf "spark.eventLog.enabled=true" \
  --conf "spark.eventLog.compress=true" \
  --conf "spark.eventLog.dir=${_DEFAULT_EVENT_LOG_PATH}" \
  ${_DEFAULT_JOB_JAR} -m "${_METADATA_PATH}"  -r "${_REVIEWS_PATH}" -c "${_CATEGORY_MAP_PATH}" -o "${_OUTPUT_PATH}" -l "${_LIMIT}" -s "${_SEED}" ${strcat}

}

###############################################################################
# Main
###############################################################################

# _main()
#
# Usage:
#   _main [<options>] [<arguments>]
#
# Description:
#   Entry point for the program, handling basic option parsing and dispatching.
_main() {
  if ((_PRINT_HELP))
  then
    _print_help
  else
    _spark_submit "$@"
  fi
}

# Call `_main` after everything has been defined.
_main "$@"
