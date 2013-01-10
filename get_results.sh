#!/bin/bash

usage()
{
    echo "usage: ${0##*/} query"
}

if [ $# -ne 1 ]
then
    usage
    exit 0
fi

mac_results=$(ssh willsky-student7.lids.mit.edu "ssh -p 22222 Alex@localhost \"cd ~/hsmm-particlefilters2/results && find . -iname '*.txt' | xargs -I{} grep -H ${1} {}\"")
ubuntu_results=$(ssh willsky-student7.lids.mit.edu "ssh -p 22225 dattalab@localhost \"cd ~/hsmm-particlefilters/results && find . -iname '*.txt' | xargs -I{} grep -H ${1} {}\"")

echo -e "$mac_results" "\n" "$ubuntu_results" | sed -e '/^\s$/d' | cat -n

echo "get one? "
read num

if [[ "$num" =~ ^[0-9]+$ ]]
then
    if [ $num -gt $(echo "$mac_results" | wc -l) ]
    then
        dirname=$(basename $(echo "$ubuntu_results" | sed -n "${num}p" | grep -Po '^\.\/[0-9\.]+'))
        filename=$(ssh willsky-student7.lids.mit.edu "ssh -p 22225 dattalab@localhost \"ls -1 ~/hsmm-particlefilters/results/${dirname}\"" | sort -n | tail -1)
        ssh willsky-student7.lids.mit.edu "rsync -qavz -e \"ssh -p 22225\" dattalab@localhost:~/hsmm-particlefilters/results/${dirname}/${filename} ./passthrough" && rsync -zqP willsky-student7.lids.mit.edu:~/passthrough ./requested_results
    else
        dirname=$(basename $(echo "$mac_results" | sed -n "${num}p" | grep -Po '^\.\/[0-9\.]+'))
        filename=$(ssh willsky-student7.lids.mit.edu "ssh -p 22222 Alex@localhost \"ls -1 ~/hsmm-particlefilters2/results/${dirname}\"" | sort -n | tail -1)
        ssh willsky-student7.lids.mit.edu "rsync -qavz -e \"ssh -p 22222\" Alex@localhost:~/hsmm-particlefilters2/results/${dirname}/${filename} ./passthrough" && rsync -zqP willsky-student7.lids.mit.edu:~/passthrough ./requested_results
    fi
    echo "see file ./requested_results for the most recent output of ${dirname}"
else
    echo "huh?"
fi
