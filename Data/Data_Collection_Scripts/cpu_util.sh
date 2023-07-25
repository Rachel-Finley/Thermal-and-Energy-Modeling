#topsnap = 'TERM=dumb top -1'
#echo "$topsnap">>test.txt

for (( i=0; ; i++ )); do
        date
        #mpstat -P ALL
	#top -1
	top -1 -b -d 1 | grep 'Cpu\|top -\|Tasks'
        sleep 1
done
