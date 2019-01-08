#!/bin/bash

# Downloads opensmile into ./opensmile/opensmile-2.3.0 
# and adds custom config files for GeMap features 50/10 ms

# Create opensmile directory
mkdir -p opensmile 

url=https://www.audeering.com
link=https://www.audeering.com/download/opensmile-2-3-0-tar-gz/?wpdmdl=4782
target=opensmile/opensmile-2.3.0.tar.gz 

echo "Downloading opensmile-2.3.0.tar.gz"
wget -O $target --referer $url $link --no-check-certificate


echo "Extracting Opensmile and removing tar"
tar -zxf $target -C opensmile/
rm $target


echo "Changing Configs"


custom_conf=opensmile_configs
opensmile_dir=opensmile/opensmile-2.3.0
conf="$opensmile_dir"/config
shared="$conf"/shared

mv $shared "$shared"_OLD
cp -r "$custom_conf"/shared $shared
cp -r "$custom_conf"/gemaps_10ms $conf
cp -r "$custom_conf"/gemaps_50ms $conf

echo "Done!"

OpensmileBIN="$opensmile_dir"/bin/linux_x64_standalone_static/SMILExtract
CONF50=gemaps_50ms/eGeMAPSv01a.conf
CONF10=gemaps_10ms/eGeMAPSv01a.conf

echo "-----------------------------------"
echo "Bin path (Linux): $OpensmileBIN"
echo "Config path gemaps 50ms: $CONF50"
echo "Config path gemaps 10ms: $CONF10"

echo
echo "Example:"
echo "/$OpensmileBIN - C $CONF50 -l 0 -I /path/to/wav -D /path/to/output"
echo
echo "For python use see:"
echo "https://github.com/ErikEkstedt/ml_reusable/blob/master/ml_reusable/data_extraction/gemaps.py"
echo "-----------------------------------"


