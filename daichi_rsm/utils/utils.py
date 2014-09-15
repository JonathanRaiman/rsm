def convert_lexicon_file_to_lexicon(path):
	reverse_lexicon = []
	lexicon = {}
	with open(path, 'r') as lexicon_file:
		for index, line in enumerate(lexicon_file):
			reverse_lexicon.append(line.rstrip())
			lexicon[line.rstrip()] = index
	return (lexicon, reverse_lexicon)