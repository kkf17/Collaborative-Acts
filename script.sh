#chmod u+x script
#!/bin/bash

##------------------------------------------------------------------------------------------------------------------
	# MANUAL:
	
	# 0) Mode 0: prepare datafile .csv
		# script.sh -m 0 
		# script.sh -m 0 -c 1  (with spell correction)
		
	# 2) Mode 2: preprocessing of data
		# script.sh -m 2 -p unlp -g 'DET...CONJ' [-n 0] -v 'bow/ww'
		# script.sh -m 2 [-c 1] -p unlp -g 'DET...CONJ' [-n 0] -v 'bow/ww' (with spell correction)
			# -c : spell and correction (OPTIONAL)
			# -p : type of preprocessing 
				# u : unique
				# n : normalization
				# p : remove ponctuation
				# s : remove stop words
				# l : lemmatization
			# -g : grammar selection
					#f. ex : 'DET CONJ SYM'
					#(see preprocessing files for much details)
			# -n : number of most predictive words (for vectorization) - useless- 
			# -v : type of vectorization
					# ww : word vectors
					# wm : mean / average of word vectors
					# wwt : word vectors with TF-IDF ponderation
		
##---------------------------------------------------------------------------------------------------------------------

#echo "input: $1"

mode0(){
	echo "Mode 0: prepare .csv files with correction $c.";
	mkdir "./data"
	for i in `seq 0 18`;
	do
		python3 step0.py -f ./collaborativeActs.csv -d $i
	done
	ls ./data > data.txt
	
}

mode0c(){
	echo "Mode 0: prepare .csv files with correction $c.";
	mkdir "./datacorrect"
	for i in `seq 0 18`;
	do
		python3 step0.py -f ./collaborativeActs.csv -d $i -c
	done
	ls ./datacorrect > datacorrect.txt
}



mode2(){
	echo "Mode 2: prepare collabacts objects with arguments $c $p $g $m $v"
	
	if [ $c = 1 ]
	then
		for f in ./datacorrect/*
		do
			python3 step2.py -f $f -p $p -g $g -m $m -v $v
		done
	fi

	if [ $c = 0 ]
	then
		for f in ./data/*
		do
			python3 step2.py -f $f -p $p -g $g -m $m -v $v
		done
	fi

	mkdir ./objects/data_c"$c"_"$p"_"$g"_"$m"_"$v"

	for f in ./objects/*
	do
  		if [ -f "$f" ]; then
    			mv "$f" ./objects/data_c"$c"_"$p"_"$g"_"$m"_"$v"
  		fi
	done

	ls ./objects/data_c"$c"_"$p"_"$g"_"$m"_"$v" > ./objects/data.txt 
	mv ./objects/data.txt ./objects/data_c"$c"_"$p"_"$g"_"$m"_"$v"

}


main(){
	echo "LOAD with $mode $c $p $g $m $v"
	if [ $mode = 0 ] && [ $c = 0 ]
	then
		mode0
	fi
	
	if [ $mode = 0 ] && [ $c = 1 ]
	then
		mode0c
	fi

	if [ $mode = 2 ]
	then
		mode2
	fi
}


c=0
p=''
g=' '
m=0
v='ww'
while getopts m:c:p:g:n:v:t: flag
do
	case "${flag}" in
		m) mode=${OPTARG};; 
		c) c=1;; 
		p) p=${OPTARG};; 
		g) g=${OPTARG};; #'DET CONJ' 
		n) m=${OPTARG};; 
		v) v=${OPTARG};;
	esac
done

main







