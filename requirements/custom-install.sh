
# Default Values :
# ----------------------------------
nlcodec_branch="mwe_schemes"
rtg_version="0.4.0"
custom_rtg=0

function usage {
	echo "Usage : $(basename $0) [-r RTG_VERSION] [-b NLCODEC_BRANCH] [-c]"
	echo "		RTG_VERSION = { 0.4.0 , 0.6.0 }"
	echo "		NLCODEC_BRANCH = [ mwe_schemes , decode_hack , ext-ngram ]"
	echo " 		c : Custom branch for RTG - allows decoding of experiments as of now"
}

while getopts "hcr:b:" opt; do
	case ${opt} in
		h) usage ;;
		r) rtg_version=$OPTARG;;
		b) nlcodec_branch=$OPTARG;;
		c) custom_rtg=1;;
	esac
done


echo "Nlcodec Branch : ${nlcodec_branch}"
echo "Use Custom RTG : ${custom_rtg}"
echo "RTG_VERSION : ${rtg_version}"

# Installing nlcodec 
# ----------------------------------
nlcodec_string="git+https://github.com/pegasus-lynx/nlcodec.git@${nlcodec_branch}#egg=nlcodec"
pip install -e $nlcodec_string


# Installing RTG
# ----------------------------------
if [[ $custom_rtg -eq 1 ]]; then
	rtg_string="git+https://github.com/pegasus-lynx/rtg.git@version-040#egg=rtg"
	pip install -e ${rtg_string}
else
	pip install "rtg==${rtg_version}"
fi
