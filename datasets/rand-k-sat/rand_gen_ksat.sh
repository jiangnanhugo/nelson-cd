#!/usr/bin/zsh
K=5
RV=700
C=700
mkdir ${K}_${RV}_${C}
for i in {0001..1000}
do
	echo "idx $i"
	cnfgen randkcnf $K $RV $C > ${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
done
