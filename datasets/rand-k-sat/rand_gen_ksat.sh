#!/usr/bin/zsh
K=5

for size in 5000
do
	RV=${size}
	C=${size}
	mkdir ${K}_${RV}_${C}
	# install from: https://github.com/MassimoLauria/cnfgen
	for i in {0001..0100}
	do
		echo "idx $i"
		cnfgen randkcnf $K $RV $C > ${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
	done
done
