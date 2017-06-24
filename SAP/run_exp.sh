#!/bin/bash

output="exp.txt"
rm ${output}

for ((i=100000; i<1000000; i+=100000))
do
	echo "" >> ${output}
	echo "N = ${i}" >> ${output}
	echo "N = ${i}"
	./main $i >> ${output}
done

for ((i=1000000; i<=2000000; i+=500000))
do
	echo "" >> ${output}
	echo "N = ${i}" >> ${output}
	echo "N = ${i}"
	./main $i >> ${output}
done
