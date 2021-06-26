# Summary

Universal Dependencies version of syntax annotations from the GUM corpus (https://corpling.uis.georgetown.edu/gum/)

# Introduction

GUM, the Georgetown University Multilayer corpus, is an open source collection of richly annotated web texts from multiple text types. The corpus is collected and expanded by students as part of the curriculum in the course LING-367 "Computational Corpus Linguistics" at Georgetown University. The selection of text types is meant to represent different communicative purposes, while coming from sources that are readily and openly available (usually Creative Commons licenses), so that new texts can be annotated and published with ease.

The dependencies in the corpus were originally annotated using Stanford Typed Depenencies (de Marneffe & Manning 2013) and converted automatically to UD using DepEdit (https://corpling.uis.georgetown.edu/depedit/). The rule-based conversion takes into account gold entity annotations found in other annotation layers of the GUM corpus (e.g. entity annotations). The conversion script used can found in the GUM build bot code, available from the (non-UD) GUM repository. For more details see the [corpus website](https://corpling.uis.georgetown.edu/gum/).

# Documents and splits

The training, development and test sets contain complete, contiguous documents, balanced for genre. Test and dev contain similar amounts of data, usually around 2,500 tokens in each genre in each, and the rest is assigned to training. There is currently less data in the genres fiction, bio and academic, but equal amounts of these are in test and dev (about one document each) and the rest is in training data. For the exact file lists in each split see:

https://github.com/UniversalDependencies/UD_English-GUM/tree/master/not-to-release/file-lists

# Acknowledgments

GUM annotation team (so far - thanks for participating!)

Adrienne Isaac, Akitaka Yamada, Amani Aloufi, Amelia Becker, Andrea Price, Andrew O'Brien, Anna Runova, Anne Butler, Arianna Janoff, Ayan Mandal, Brandon Tullock, Brent Laing, Candice Penelton, Chenyue Guo, Colleen Diamond, Connor O'Dwyer, Dan Simonson, Didem Ikizoglu, Edwin Ko, Emily Pace, Emma Manning, Ethan Beaman, Han Bu, Hang Jiang, Hanwool Choe, Hassan Munshi, Ho Fai Cheng, Jakob Prange, Jehan al-Mahmoud, Jemm Excelle Dela Cruz, Joaquin Gris Roca, John Chi, Jongbong Lee, Juliet May, Katarina Starcevic, Katherine Vadella, Lara Bryfonski, Lindley Winchester, Logan Peng, Lucia Donatelli, Margaret Anne Rowe, Margaret Borowczyk, Maria Stoianova, Mariko Uno, Mary Henderson, Maya Barzilai, Md. Jahurul Islam, Michaela Harrington, Minnie Annan, Mitchell Abrams, Mohammad Ali Yektaie, Naomee-Minh Nguyen, Nicholas Workman, Nicole Steinberg, Rachel Thorson, Rebecca Childress, Ruizhong Li, Ryan Murphy, Sakol Suethanapornkul, Sean Macavaney, Sean Simpson, Shannon Mooney, Siddharth Singh, Siyu Liang, Stephanie Kramer, Sylvia Sierra, Timothy Ingrassia, Wenxi Yang, Xiaopei Wu, Yang Liu, Yilun Zhu, Yingzhu Chen, Yiran Xu, Young-A Son, Yushi Zhao, Zhuxin Wang, Amir Zeldes

... and other annotators who wish to remain anonymous!  

## References

As a scholarly citation for the corpus in articles, please use this paper:

* Zeldes, Amir (2017) "The GUM Corpus: Creating Multilayer Resources in the Classroom". Language Resources and Evaluation 51(3), 581–612.

<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.2
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: academic fiction nonfiction news spoken web wiki
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: converted from manual
Contributors: Peng, Siyao;Zeldes, Amir
Contributing: elsewhere
Contact: amir.zeldes@georgetown.edu
===============================================================================
</pre>
