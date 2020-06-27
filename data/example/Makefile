download-glove:
	#wget -P . "http://nlp.stanford.edu/data/glove.840B.300d.zip"
	axel "http://nlp.stanford.edu/data/glove.840B.300d.zip" --output=$(DATA)
	unzip $(DATA)/glove.840B.300d.zip -d $(DATA)/glove.840B.300d.txt
	ln -s $(DATA)/glove.840B.300d.txt/glove.840B.300d.txt ./glove.840B.300d.txt
	rm $(DATA)/glove.840B.300d.zip

download-CoNLL-2003:
	#https://www.clips.uantwerpen.be/conll2003/ner/
	axel "http://nlp.stanford.edu/data/glove.840B.300d.zip" --output=$(DATA)
	unzip $(DATA)/glove.840B.300d.zip -d $(DATA)/glove.840B.300d.txt
	ln -s $(DATA)/glove.840B.300d.txt/glove.840B.300d.txt ./glove.840B.300d.txt
	rm $(DATA)/glove.840B.300d.zip
