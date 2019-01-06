#!/bin/bash

# Downloads opensmile into ./opensmile

# Create opensmile directory
mkdir -p opensmile 

url=https://www.audeering.com
link=https://www.audeering.com/download/1318/
target=opensmile/opensmile-2.3.0.tar.gz 

echo "Downloading opensmile-2.3.0.tar.gz"
wget -O $target --referer $url $link 


echo "Extracting Opensmile and removing tar"
tar -zxf $target -C opensmile/
rm $target


echo "Changing Configs"

conf=opensmile/opensmile-2.3.0/config
shared="$conf"/shared
custom_conf=opensmile/configs

mv $shared "$shared"_OLD
cp -r "$custom_conf"/shared $shared
cp -r "$custom_conf"/gemaps_10ms $conf
cp -r "$custom_conf"/gemaps_50ms $conf

echo "Done!"
