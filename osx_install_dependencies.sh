#!/bin/bash   

# Text colors
# E.g. printf "${bldred}Hello World{txtrst}\n"
txtred='\e[0;31m' # Red
txtgrn='\e[0;32m' # Green
txtylw='\e[0;33m' # Yellow
txtblu='\e[0;34m' # Blue
txtpur='\e[0;35m' # Purple
txtcyn='\e[0;36m' # Cyan
txtwht='\e[0;37m' # White
bldblk='\e[1;30m' # Black - Bold
bldred='\e[1;31m' # Red
bldgrn='\e[1;32m' # Green
bldylw='\e[1;33m' # Yellow
bldblu='\e[1;34m' # Blue
bldpur='\e[1;35m' # Purple
bldcyn='\e[1;36m' # Cyan
bldwht='\e[1;37m' # White
unkblk='\e[4;30m' # Black - Underline
undred='\e[4;31m' # Red
undgrn='\e[4;32m' # Green
undylw='\e[4;33m' # Yellow
undblu='\e[4;34m' # Blue
undpur='\e[4;35m' # Purple
undcyn='\e[4;36m' # Cyan
undwht='\e[4;37m' # White
bakblk='\e[40m'   # Black - Background
bakred='\e[41m'   # Red
bakgrn='\e[42m'   # Green
bakylw='\e[43m'   # Yellow
bakblu='\e[44m'   # Blue
bakpur='\e[45m'   # Purple
bakcyn='\e[46m'   # Cyan
bakwht='\e[47m'   # White
txtrst='\e[0m'    # Text Reset
txtblk='\e[0;30m' # Black - Regular
txtred='\e[0;31m' # Red
txtgrn='\e[0;32m' # Green
txtylw='\e[0;33m' # Yellow
txtblu='\e[0;34m' # Blue
txtpur='\e[0;35m' # Purple
txtcyn='\e[0;36m' # Cyan
txtwht='\e[0;37m' # White
bldblk='\e[1;30m' # Black - Bold
bldred='\e[1;31m' # Red
bldgrn='\e[1;32m' # Green
bldylw='\e[1;33m' # Yellow
bldblu='\e[1;34m' # Blue
bldpur='\e[1;35m' # Purple
bldcyn='\e[1;36m' # Cyan
bldwht='\e[1;37m' # White
unkblk='\e[4;30m' # Black - Underline
undred='\e[4;31m' # Red
undgrn='\e[4;32m' # Green
undylw='\e[4;33m' # Yellow
undblu='\e[4;34m' # Blue
undpur='\e[4;35m' # Purple
undcyn='\e[4;36m' # Cyan
undwht='\e[4;37m' # White
bakblk='\e[40m'   # Black - Background
bakred='\e[41m'   # Red
bakgrn='\e[42m'   # Green
bakylw='\e[43m'   # Yellow
bakblu='\e[44m'   # Blue
bakpur='\e[45m'   # Purple
bakcyn='\e[46m'   # Cyan
bakwht='\e[47m'   # White
txtrst='\e[0m'    # Text Reset

# do's list contain value?
# E.g.: contains aList anItem
contains() {
	[[ $1 =~ $2 ]] && echo true || echo false
}

# is homebrew package installed?
# E.g.: is_installed package
#
# Check if Qt package is installed:
#	if [ "$(is_installed qt)"=false ]; then
#		echo "not installed"
#	fi
#
brew_is_installed() {
	echo "$(contains "$(brew list)" $1)"
}

# call 'brew install', but only when it is not already installed with the specified options
# E.g.: brew_install_smart qt --with-developer
brew_install_smart() {
	if (( $# >= 1 )) ; then
		INFO="$(brew info $1)"
		if [[ -z "$INFO" ]] ; then
		  printf "${bldred}Package '$1' not available from homebrew{txtrst}\n"
		  echo false
		else
			INSTALL=false
			INSTALLED=false
			if $(brew_is_installed $1) ; then
				INSTALLED=true
				IFS=$'\n' read -rd '' -a lines <<< "$INFO"
				for line in "${lines[@]}" ; do
					if [[ $line == *"Built from source with"* ]] ; then		    	
				    	IFS=$':' read -rd '' -a line <<< "$line"
				    	if (( ${#line[@]} >= 2 )) ; then
					    	IFS=$', ' read -rd '' -a options <<< "${line[1]}"
				    		FIRST=true
					    	for var in "$@" ; do
					    		if $FIRST ; then
					    			FIRST=false
					    		else
						    		if ! "$(contains "${line[1]}" "$var")" ; then
											INSTALL=true
									fi
								fi
							done
				    	fi
					fi
				done
			else
				INSTALL=true
			fi

			if $INSTALL ; then
				if $INSTALLED ; then
					CMD="brew uninstall $1"
					eval "$CMD"
				fi
				CMD="brew install"
				for var in "$@" ; do
					CMD="$CMD $var"
				done
				eval "$CMD"

				CMD="brew link $1 --overwrite"
				eval "$CMD"

				brew linkapps
			else
				printf "${txtgrn}$1 already installed${txtrst}\n"
			fi	
		fi
	fi
}

unamestr=`uname`
if [[ "$unamestr" == 'Darwin' ]]; then
	printf "${txtpur}Installing dependencies for itom${txtrst}\n"
	echo " "
	echo " "
	echo "This script will install all necessary dependencies for itom on OS X."
	echo "It is using homebrew as its package manager and pip for installing python packages."
	echo "To be able to use python 3, which is required when using itom, we will set aliases for pithon and pip."
	echo "This means when you are using the 'python' command in a command line, it will from now on call python3."
	echo "This will not effect applications that depend on python 2!"
	echo " "
	echo " "
	printf "${bldred}ATTENTION:${txtrst}\n"
	printf "${txtred}  Xcode along with its command line utilities must be installed${txtrst}\n"
	printf "${txtred}  Cmake must already be installed (https://www.cmake.org)${txtrst}\n"
	echo " "

	echo "-----"
	printf "${txtcyn}Do you meet the requirements and are ready to go (y/n)?${txtrst}\n"
	printf "This might take a while."
	read answer
	if echo "$answer" | grep -iq "^y" ;then
	    echo " "
	else
	    return
	fi

	echo "-----"
	printf "${txtcyn}Do you want to set aliases for python3 so that the command python will call python3 (y/n)?${txtrst}\n"
	printf "Only skip this when you know what you are doing."
	read answer2

	echo " "
	echo "-----"
	printf "${txtcyn}Should we create directories at ~/itom and get the source code? (y/n)?${txtrst}"
	read answer3

	echo " "
	echo "-----"
	printf "${bldgrn}Let's go!${txtrst}\n"
	printf "${txtylw}This might take a while (we are talking hours).${txtrst}\n"

	echo " "
	printf "${txtylw}Installing Homebrew if necessary${txtrst}\n"
	echo "++++++++++++++++++++++++++++++++"
	type brew >/dev/null 2>&1 || { echo >&2 "Homebrew is required but not installed.  Installing it now."; ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"; }

	echo " "
	printf "${txtylw}Adding taps${txtrst}\n"
	echo "+++++++++++"
	brew tap homebrew/science
	brew tap homebrew/python

	echo " "
	printf "${txtylw}Updating Homebrew${txtrst}\n"
	echo "+++++++++++++++++"
	sudo brew update -v

	echo " "
	printf "${txtylw}Installing kegs${txtrst}\n"
	echo "+++++++++++++++"
	brew_install_smart git
	brew_install_smart gcc 
	brew_install_smart python3 --devel

	if echo "$answer2" | grep -iq "^y" ;then
		echo " "
		printf "${txtylw}Please set the aliases for python now${txtrst}\n"
		echo " "
		echo "The default Python version on OS X is 2.x. Since |itom| is using Python 3.x you installed in the previous step but it is'nt recommended to replace version 2.x with 3.x. We will set an alias for python3, so when entered python in a terminal session, python3 will be called."
		echo "To edit you aliases execute the following commands. The same thing must be done for pip and easy_install."
		echo "Be adviced to check the installed version number of python and change it when necessary."
		echo " "
		printf "${txtpur}  printf \"alias python='python3'\n\" >> ~/.bash_profile${txtrst}\n"
		printf "${txtpur}  printf \"alias easy_install='/usr/local/Cellar/python3/3.x.y/bin/easy_install-3.x'\n\" >> ~/.bash_profile${txtrst}\n"
		printf "${txtpur}  printf \"alias pip='/usr/local/Cellar/python3/3.x.y/bin/pip3.x'\n\" >> ~/.bash_profile${txtrst}\n"
		printf "${txtpur}  . ~/.bash_profile${txtrst}\n"
		echo " "
		echo " The command python --version will give you the installed version number."
		echo " "
		read -p "Press [Enter] when you are ready ..."
	else
		echo "Skipping to set python aliases${txtrst}\n"
	fi
	. ~/.bash_profile

	brew_install_smart pkg-config
	brew_install_smart qt --with-developer --with-docs
	brew_install_smart pyqt --with-python3
	brew_install_smart qscintilla2 --with-python3
	brew_install_smart ffmpeg
	brew_install_smart glew
	brew_install_smart fftw

	echo " "
	printf "${txtylw}Installing Python packages using Homebrew${txtrst}\n"
	echo "+++++++++++++++++++++++++++++++++++++++++"
	brew_install_smart numpy --with-python3
	brew_install_smart pillow --with-python3
	brew_install_smart matplotlib --with-python3 --with-pyqt
	brew_install_smart matplotlib-basemap --with-python3
	brew_install_smart scipy --with-python3
	brew_install_smart opencv 
	brew_install_smart pcl 
	brew_install_smart caskroom/cask/brew-cask
	brew cask install qt-creator
	brew linkapps

	echo " "
	printf "${txtylw}Installing Python packages using setuptools${txtrst}\n"
	echo "+++++++++++++++++++++++++++++++++++++++++++"
	sudo easy_install virtualenv

	echo " "
	printf "${txtylw}Installing Python packages using PIP${txtrst}\n"
	echo "++++++++++++++++++++++++++++++++++++"
	pip install pyparsing frosted ipython mpmath sphinx

	if echo "$answer3" | grep -iq "^y" ;then
		echo " "
		printf "${txtylw}Creating directories${txtrst}\n"
		echo "++++++++++++++++++++"
		mkdir ~/itom
		mkdir ~/itom/sources
		mkdir ~/itom/build_debug
		mkdir ~/itom/build_debug/itom
		mkdir ~/itom/build_debug/plugins
		mkdir ~/itom/build_debug/designerPlugins
		mkdir ~/itom/build_release
		mkdir ~/itom/build_release/itom
		mkdir ~/itom/build_release/plugins
		mkdir ~/itom/build_release/designerPlugins	
		cd ~/itom/sources
		git clone https://bitbucket.org/itom/itom.git
		git clone https://bitbucket.org/itom/plugins.git
		git clone https://bitbucket.org/itom/designerplugins.git	
	fi

	echo " "
	echo " "
	printf "${bldgrn}Finished. Open up Cmake and get compiling${txtrst}\n"
	if echo "$answer3" | grep -iq "^y" ;then
		echo "you will find itom in the directory '~/itom' in your user directory"
	fi
	echo "Have fun with itom"
else
	echo "Error: This script is designed for OS X only."
fi