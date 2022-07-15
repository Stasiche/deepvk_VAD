with open('librispeech-lexicon.txt', 'r') as f:
    with open('tabed_lexion.txt', 'w') as out:
        for line in f.readlines():
            line = line.split()
            out.write(f'{line[0]}\t' + ' '.join(line[1:])+'\n')