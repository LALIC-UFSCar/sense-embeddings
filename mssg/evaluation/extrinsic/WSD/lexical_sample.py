import sensegram
from wsd import WSD
from gensim.models import KeyedVectors
from metrics import Metrics

#ambiguous_word wordnet_sense|cod_sense_vector
ambiguous_word = []
ambiguous_word.append(('obra', 575741, 3)) #reforma, construção
ambiguous_word.append(('obra', 3841417, 4)) #pintura, quadro
ambiguous_word.append(('centro', 2993546, 3)) #instituição
ambiguous_word.append(('centro', 8523483, 2)) #local na área central
ambiguous_word.append(('estado', 24720, 4)) #modo, situação, estado de emergência, estado de saúde
ambiguous_word.append(('estado', 8654360, 3)) #estados de um país
ambiguous_word.append(('presidente', 10467179, 2)) #presidente do Brasil
ambiguous_word.append(('presidente', 10467395, 3)) #presidente de outros países
ambiguous_word.append(('presidente', 10468962, 4)) #presidente de outros órgãos
ambiguous_word.append(('discurso', 7238694, 3)) #discurso como ideias
ambiguous_word.append(('discurso', 7109196, 4)) #discurso específico feito em algum lugar
ambiguous_word.append(('salto', 7414222, 3)) #salto em reais
ambiguous_word.append(('salto', 120515, 4)) #salto em metros (pulo)
ambiguous_word.append(('escola', 4511002, 2)) #escola de ensino
ambiguous_word.append(('escola', 8275185, 4)) #escola de samba
ambiguous_word.append(('sexta', 13847402, 3)) #sexta colocação
ambiguous_word.append(('sexta', 15164463, 4)) #sexta-feira
ambiguous_word.append(('pontos', 8620061, 4)) #pontos de alagamento
ambiguous_word.append(('pontos', 13610162, 3)) #pontos percentuais
ambiguous_word.append(('anos', 4924103, 3)) #anos: idade
ambiguous_word.append(('anos', 15203791, 4)) #anos: nos últimos anos...

sense_vectors_fpath = "../../../../models/multisense_s300_ptbr_sg.sense_vectors"
word_vectors_fpath = "../../../../models/word2vec_s300_ptbr_sg.txt"
sv = sensegram.SenseGram.load_word2vec_format(sense_vectors_fpath, binary=False)
wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False, unicode_errors="ignore")


def get_mfs(arrTrue, mfs_sense):
    arrPred = []
    for i in arrTrue:
        arrPred.append(mfs_sense)

    print('=====================MFS=======================================')
    print('arrTrue: ', arrTrue)
    print('arrPred: ', arrPred)
    accuracy = Metrics(arrTrue, arrPred).accuracy()
    f1Score = Metrics(arrTrue, arrPred).f1_score()
    precision = Metrics(arrTrue, arrPred).precision_score()

    print('accuracy: ', accuracy * 100)
    print('f1Score: ', f1Score * 100)
    print('precision: ', precision * 100)

def get_metrics(arrTrue, arrPred):
    accuracy = Metrics(arrTrue, arrPred).accuracy()
    f1Score = Metrics(arrTrue, arrPred).f1_score()
    precision = Metrics(arrTrue, arrPred).precision_score()

    print('accuracy: ', accuracy * 100)
    print('f1Score: ', f1Score * 100)
    print('precision: ', precision * 100)
    #return 'Accuracy: ', accuracy, 'f1Score: ', f1Score


def get_neighbor(word):
    for sense_id, prob in sv.get_senses(word):
        print(sense_id)
        print("="*20)
        for rsense_id, sim in sv.wv.most_similar(sense_id, topn=30):
            print("{} {:f}".format(rsense_id, sim))
        print("\n")


def get_sensevector(sense):
    s = 0
    for i in ambiguous_word:
        if str(i[1]) == str(sense):
            s = i[2]
    return s


def get_wsd(word, context):
    context_words_max = 5 # change this paramters to 1, 2, 5, 10, 15, 20 : it may improve the results
    context_window_size = 10 # this parameters can be also changed during experiments
    ignore_case = True
    method = 'sim'
    lang = "pt" # to filter out stopwords

    # Disambiguate a word in a context
    wsd_model = WSD(sv, wv, window=context_window_size, lang=lang,
                    max_context_words=context_words_max, ignore_case=ignore_case, method=method, verbose=False)
    return wsd_model.disambiguate(context, word)


def get_lexicalsample(ambiguous_word):
    arrTrue = []
    arrPred = []
    pred_s = []
    mfs = 0
    qtde_sentence = 0
    mfs_sense = 0
    with open('../../../../datasets/CSTNews.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        for line in f:
            elements = line.split(' ')
            if len(elements) == 3:
                word = elements[0]
                sense = elements[1].replace('<','').replace('>','')
                true_s = 0
                qtde = 0
            else:
                if word == ambiguous_word:
                    if len(elements) > 3:
                        sentence = line.replace('<' + sense + '>','')
                        pred_sense = get_wsd(word, sentence)
                        pred_s = pred_sense[0].split('#')
                        true_s = get_sensevector(sense)
                        if true_s > 0:
                            arrTrue.append(true_s)
                            arrPred.append(int(pred_s[1]))
                            qtde = qtde + 1
                            qtde_sentence = qtde_sentence + 1
                            if qtde > mfs:
                                mfs = qtde
                                mfs_sense = true_s

    print('==================sense vector=========================================')
    print('arrTrue: ', arrTrue)
    print('arrPred: ', arrPred)
    get_metrics(arrTrue, arrPred)
    get_mfs(arrTrue, mfs_sense)


def get_ambiguousword():
    aword = []
    for i in ambiguous_word:
        aword.append(i[0])
    words = list(set(aword))
    for i in words:
        print('===============================================================')
        print('Ambiguous word: ', i)
        get_lexicalsample(i)

get_neighbor('acidente')
get_neighbor('aumento')

#get_lexicalsample('centro')
#get_ambiguousword()

#python lexical_sample.py
