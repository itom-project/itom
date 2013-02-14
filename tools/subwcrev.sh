#! /bin/bash
# This is a script to emulate the function of subwcrev.exe which is part of subversion
# installation on windows but not on unix.
# Generate version.h from version.in
# Written by Dr. Georg Wiora
# 06-Apr-2012

# LEGAL:
# This file is put under the LGPL 1.0 or newer license and belongs to the openGPS project.
# Use and distribution independently of openGPS is explicitly allowed, as long
# as this note and the original authors name is kept.

#set -x

# Check number of parameters
if [ -z $3 ]
then
  echo "Usage: subwcrev.sh {Project Dir} {source.in} {source.out}
  This command will check the subversion state of Project dir, read in the template file \"source.in\"
  and generate the output file \"source.out\"."
  exit 1
fi

#set
# Base dir of project
based="$1"
# input template to generate file from
SCRIPT_INPUT_FILE_0="$2"
# final output file
SCRIPT_OUTPUT_FILE_1="$3"

# Get current version of base dir
ver=`svnversion -n "${based}"`

# Check for modifications in Working copy
# mod is zero if version string does not contain an "M" else 1
mod=$(( 1 - `echo $ver | grep -c -i "M"` ))

# create string if modified
if [ $mod -eq 0 ]
then
  modstr="\"Warning! Contains locally modified code!\""
else
  modstr="\"\""
fi


# Check for mixed versions. Version string will contain a colon ":"
mixed=$(( 1 - `echo $ver | grep -c ":"` ))

# create string if mixed versions
if [ $mixed -eq 0 ]
then
  mixstr="\"Warning! Contains mixed revisions!\""
else
  mixstr="\"\""
fi


# get current time stamp
now=`date`

# Get Last change date. 
# BUG: This is not the date of the revision but of the folders revision.
lastchangedate=`svn info "${based}" |
grep -i "Last Changed Date:" |
cut '-d ' -f "4-"`

# Get repository url
repurl=`svn info "${based}" |
grep -i "URL:" |
cut '-d ' -f "2-" |
sed -e 's/\\//\\\\\\//g'`

if [ -f "$SCRIPT_INPUT_FILE_0" ]
then
 echo -n ""
else
  echo "Error: Source file \"$SCRIPT_INPUT_FILE_0\" does not exist!"
  exit 1
fi


echo "Creating file \"${SCRIPT_OUTPUT_FILE_1}\""
echo "Version: $ver"
echo "Date: $lastchangedate"
echo "Clean Revision Flag: $mixed (==0 for mixed revisions)"
echo "Clean Build Flag: $mod (==0 for local modifications)"


cat "$SCRIPT_INPUT_FILE_0" |
sed -e "s/\\\$WCRANGE\\\$/${ver}/g" |
sed -e "s/\\\$WCDATE\\\$/${lastchangedate}/g" |
sed -e "s/\\\$WCURL\\\$/${repurl}/g" |
sed -e "s/\\\$WCMODS?\".*\$/${modstr}/g" |
sed -e "s/\\\$WCMODS?0.*\$/${mod}/g" |
sed -e "s/\\\$WCMIXED?\".*\$/${mixstr}/g" |
sed -e "s/\\\$WCMIXED?0.*\$/${mixed}/g" |
sed -e "s/\\\$WCNOW\\\$/${now}/g" >"$SCRIPT_OUTPUT_FILE_1"
