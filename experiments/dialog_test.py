import sys
import csv
import nltk
import regex
import pickle
import shelve
import logging
import pymystem3
import itertools
import collections
import numpy as np
import pandas as pd

from tqdm import tqdm
from unionfind.unionfind import UnionFind
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

# TODO
data_dir = sys.argv[1]
tasks_dir = data_dir + "tasks/"

# Word tokenization.

token_regexp = regex.compile("(?u)\\b(\\p{L}+|\d+)\\b")

def tokenize(text):
    return token_regexp.findall(text)

# Sentence tokenization.

sent_detector = nltk.data.load("russian.pickle")
max_short_len = 3

def sent_tokenize(text):
    # Basic preprocessing.
    text = text.lower()

    # Step 1: generate sentences and spans.
    sentences = [tokenize(x) for x in sent_detector.tokenize(text)]
    spans = sent_detector.span_tokenize(text)

    # Step 2: concatenate short sentences.
    out_sentences = []
    out_spans = []
    short_sentence = []
    short_span = None
    for sent, span in zip(sentences, spans):
        if short_span:
            short_sentence += sent
            short_span = (short_span[0], span[1])
        elif len(sent) <= max_short_len:
            short_sentence = sent
            short_span = span
        if len(sent) > max_short_len:
            if short_span:
                out_sentences.append(short_sentence)
                out_spans.append(short_span)
                short_sentence = []
                short_span = None
            else:
                out_sentences.append(sent)
                out_spans.append(span)
    if short_span:
        out_sentences.append(short_sentence)
        out_spans.append(short_span)

    return out_sentences, out_spans

mystem_to_uni_str = """
A       ADJ
ADV     ADV
ADVPRO  ADV
ANUM    ADJ
APRO    DET
COM     ADJ
CONJ    SCONJ
INTJ    INTJ
NONLEX  X
NUM     NUM
PART    PART
PR      ADP
S       NOUN
SPRO    PRON
UNKN    X
V       VERB
"""

mystem_to_uni_map = dict(map(str.split, mystem_to_uni_str.strip().split("\n")))

stemmer = pymystem3.Mystem()

add_unparsed = True
gr_regexp = regex.compile("[^\w]")

def lemmatize(tokens):
    lemmas = []
    tokens_str = " ".join(tokens)
    for res in stemmer.analyze(tokens_str):
        if res.get("analysis"):
            info = res["analysis"][0]
            stem_pos, *_ = gr_regexp.split(info["gr"].upper())
            lemmas.append("%s_%s" % (info["lex"].strip(), mystem_to_uni_map.get(stem_pos, "X")))
            #lemmas.append(info["lex"].strip())
        elif add_unparsed:
            lemmas.append(res["text"].strip())
    return list(filter(None, lemmas))

def strip_pos(lemma):
    return lemma.split("_", 1)[0]

def intersects(a1, b1, a2, b2):
    return b1 >= a2 and b2 >= a1

def grouper(iterable, n):
    sourceiter = iter(iterable)

    while True:
        batchiter = itertools.islice(sourceiter, n)
        yield tuple(itertools.chain([next(batchiter)], batchiter))

def question_to_vec(lemmas, embeddings, zero_vec, weights=None, normalize_weights=True):
    if weights is None:
        weights = {}
    words = list(filter(embeddings.__contains__, lemmas))
    vs = [zero_vec] + list(map(embeddings.__getitem__, words))
    ws = np.array([0.] + list(map(lambda w: weights.get(w, 1.), words)))
    ws = ws.reshape((len(ws), 1))
    if normalize_weights:
        ws = normalize(ws, norm="l1", axis=0)
    vec = np.sum(ws * vs, axis=0)
    return vec

def get_embedding(sent_id, embeddings):
    _, i = sentences_files_ids[sent_id]
    return embeddings[i]

logging.info("Start bootstrapping.")

# Construct paths lists.

paths = collections.OrderedDict()

with open(tasks_dir + "pairs") as fin:
    for line in fin:
        susp_name, src_name = line.strip().split()
        paths["susp/" + susp_name] = 1
        paths["src/" + src_name] = 1

paths = list(paths.keys())

# Read the vocabulary.

logging.info("Read the vocabulary.")

lvocab = set()

for path in tqdm(paths):
    with open(data_dir + path) as fin:
        fin_text = fin.read()
        for tokens, _ in zip(*sent_tokenize(fin_text)):
            lemmas = lemmatize(tokens)
            lvocab.update(lemmas)

vocab = dict(zip(map(strip_pos, lvocab), range(len(lvocab))))
lvocab = dict(zip(lvocab, range(len(lvocab))))

sentences_dict = {}
files_sentences_ids = {}
sentences_files_ids = []
sents_cnt = 0

logging.info("Construct TF-IDF vectors.")

# For TF-IDF.
sentences_lemmas = []
sentences_llemmas = []

with shelve.open("sentences") as sentences_dict:
    for path in tqdm(paths):
        with open(data_dir + path) as fin:
            sents_dict = {}
            files_sentences_ids[path] = {}
            fin_text = fin.read()
            for i, (tokens, _) in enumerate(zip(*sent_tokenize(fin_text))):
                if tokens:
                    lemmas_ = lemmatize(tokens)
                    lemmas = np.array(list(map(lambda x: vocab[strip_pos(x)], lemmas_)), dtype=np.int32)
                    llemmas = np.array(list(map(lambda x: lvocab[x], lemmas_)), dtype=np.int32)
                    files_sentences_ids[path][i] = sents_cnt
                    sentences_files_ids.append((path, i))
                    sents_dict[i] = (lemmas, llemmas)
                    sentences_lemmas.append(lemmas)
                    sentences_llemmas.append(llemmas)
                    sents_cnt += 1
            sentences_dict[path] = sents_dict

# Read RusVectores word embeddings.

logging.info("Read RV embeddings.")

rv_full_embeddings = KeyedVectors.load_word2vec_format("ruwikiruscorpora_0_300_20.bin", binary=True)

rv_dict = dict(zip(rv_full_embeddings.index2word, range(len(rv_full_embeddings.index2word))))
rv_embeddings = {v: rv_full_embeddings.syn0[k] for v, k in rv_dict.items() if v in lvocab}

del rv_full_embeddings, rv_dict

tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, binary=True)

tfidf.fit(sentences_llemmas)
inv_ldict = dict(map(reversed, tfidf.vocabulary_.items()))

del sentences_llemmas

# Make RusVectores weighted sentence embeddings.

logging.info("Make RV sent embeddings.")

rv_zero_vec = np.zeros(300)
lvocab_inv = dict(map(reversed, lvocab.items()))

with shelve.open("sentences") as sentences_dict, shelve.open("rusvectores") as rv_embeddings_dict:
    for path in tqdm(paths):
        sents_dict = sentences_dict[path]
        rv_embeds = []
        sents = list(map(lambda x: x[1][1], sorted(sents_dict.items())))
        sents_tfidfs = tfidf.transform(sents)
        for i in range(len(sents)):
            sent = list(map(lvocab_inv.__getitem__, sents[i]))
            row = sents_tfidfs.getrow(i)
            _, ix = row.nonzero()
            words = list(map(lambda x: lvocab_inv[inv_ldict[x]], ix))
            weights = dict(zip(words, row.data))
            rv_embeds.append(question_to_vec(sent, rv_embeddings, rv_zero_vec, weights))
        rv_embeddings_dict[path] = np.array(rv_embeds, dtype=np.float32)

del rv_embeddings, tfidf, inv_ldict, lvocab_inv

tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, binary=True)

tfidf.fit(sentences_lemmas)
inv_dict = dict(map(reversed, tfidf.vocabulary_.items()))

del sentences_lemmas

# Read fastText word embeddings.

logging.info("Read FT embeddings.")

ft_embeddings = {}

with open("dialog.fasttext.vec") as fin:
    n_lines, dim = map(int, fin.readline().split())
    for line in tqdm(fin):
        word, *vec = line.strip().split(" ")
        if word in vocab:
            vec = np.array(list(map(float, vec)))
            ft_embeddings[word] = vec

# Make fastText weighted sentence embeddings.

logging.info("Make FT sent embeddings.")

ft_zero_vec = np.zeros(100)
vocab_inv = dict(map(reversed, vocab.items()))

with shelve.open("sentences") as sentences_dict, shelve.open("fasttext") as ft_embeddings_dict:
    for path in tqdm(paths):
        sents_dict = sentences_dict[path]
        ft_embeds = []
        sents = list(map(lambda x: x[1][0], sorted(sents_dict.items())))
        sents_tfidfs = tfidf.transform(sents)
        for i in range(len(sents)):
            sent = list(map(vocab_inv.__getitem__, sents[i]))
            row = sents_tfidfs.getrow(i)
            _, ix = row.nonzero()
            words = list(map(lambda x: vocab_inv[inv_dict[x]], ix))
            weights = dict(zip(words, row.data))
            ft_embeds.append(question_to_vec(sent, ft_embeddings, ft_zero_vec, weights))
        ft_embeddings_dict[path] = np.array(ft_embeds, dtype=np.float32)

del ft_embeddings, vocab_inv

# Read Starspace embeddings.

logging.info("Read SS embeddings.")

ss_embeddings = {}

with open("dialog.starspace.train.tsv") as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    for row in tqdm(tsv_reader):
        word, *vec = row
        if word in vocab:
            vec = np.array(list(map(float, vec)))
            ss_embeddings[word] = vec

# Make Starspace weighted sentence embeddings.

logging.info("Make SS sent embeddings.")

ss_zero_vec = np.zeros(100)
vocab_inv = dict(map(reversed, vocab.items()))

with shelve.open("sentences") as sentences_dict, shelve.open("starspace") as ss_embeddings_dict:
    for path in tqdm(paths):
        sents_dict = sentences_dict[path]
        ss_embeds = []
        sents = list(map(lambda x: x[1][0], sorted(sents_dict.items())))
        sents_tfidfs = tfidf.transform(sents)
        for i in range(len(sents)):
            sent = list(map(vocab_inv.__getitem__, sents[i]))
            row = sents_tfidfs.getrow(i)
            _, ix = row.nonzero()
            words = list(map(lambda x: vocab_inv[inv_dict[x]], ix))
            weights = dict(zip(words, row.data))
            ss_embeds.append(question_to_vec(sent, ss_embeddings, ss_zero_vec, weights))
        ss_embeddings_dict[path] = np.array(ss_embeds, dtype=np.float32)

del ss_embeddings, vocab_inv
del tfidf, inv_dict

logging.info("Load classifier.")

with open("classifier2.0_dialog_all.mdl", "rb") as fin:
    clf = pickle.load(fin)

top_N = 5

# Create dataset out of embeddings ranking.

pre_detections_list = []

test_paths = []

logging.info("Start prediction.")

with shelve.open("rusvectores") as rv, shelve.open("fasttext") as ft, shelve.open("starspace") as ss, \
     shelve.open("sentences") as sentences_dict:
    embeddings_list = (rv, ft, ss)
    with open(tasks_dir + "pairs") as fin:
        for line in tqdm(fin):
            susp_name, src_name = line.strip().split()
            susp_path, src_path = "susp/" + susp_name, "src/" + src_name
            test_paths.append((susp_path, src_path))
            susp_sents_is = list(files_sentences_ids[susp_path].keys())
            susp_sents_ids = list(files_sentences_ids[susp_path].values())
            src_sents_is = list(files_sentences_ids[src_path].keys())
            src_sents_ids = np.array(list(files_sentences_ids[src_path].values()))
            cands_sents_ids = collections.defaultdict(set)
            features, indices, pre_detections = [], [], []
            sents_susp_dict, sents_src_dict = sentences_dict[susp_path], sentences_dict[src_path]
            # Шаг 1: формирование кандидатов
            for embeddings in embeddings_list:
                susp_embeddings = embeddings[susp_path]
                src_embeddings = embeddings[src_path]
                susp_vecs = list(map(lambda x: get_embedding(x, susp_embeddings), susp_sents_ids))
                src_vecs = list(map(lambda x: get_embedding(x, src_embeddings), src_sents_ids))
                dists = (1 - cosine_similarity(susp_vecs, src_vecs))
                top_ranks_list = np.argsort(dists, axis=1)[:, :top_N]
                for susp_sent_id, top_ranks in zip(susp_sents_ids, top_ranks_list):
                    cands_sents_ids[susp_sent_id].update(src_sents_ids[top_ranks])
            for susp_sent_i, susp_sent_id in zip(susp_sents_is, susp_sents_ids):
                susp_lemmas = sents_susp_dict[susp_sent_i][1]
                susp_lemmas_set = set(tuple(susp_lemmas))
                # Шаг 2: вычисление расстояния по всем эмбеддингам
                cand_sents_ids = np.array(list(cands_sents_ids[susp_sent_id]))
                top_indices = list(map(lambda x: (susp_sent_id, x), cand_sents_ids))
                top_dists = []
                for embeddings in embeddings_list:
                    susp_embeddings = embeddings[susp_path]
                    src_embeddings = embeddings[src_path]
                    susp_vec = get_embedding(susp_sent_id, susp_embeddings)
                    cand_vecs = list(map(lambda x: get_embedding(x, src_embeddings), cand_sents_ids))
                    dists = (1 - cosine_similarity([susp_vec], cand_vecs))[0]
                    top_dists.append(dists)
                left_incl_dists = [] # susp_lemmas \in cand_lemmas
                right_incl_dists = [] # cand_lemmas \in susp_lemmas
                susp_lens_dists = []
                cand_lens_dists = []
                min_lens_dists = []
                for cand_sent_id in cand_sents_ids:
                    cand_path, cand_sent_i = sentences_files_ids[cand_sent_id]
                    assert cand_path == src_path
                    cand_lemmas = sents_src_dict[cand_sent_i][1]
                    cand_lemmas_set = set(tuple(cand_lemmas))
                    intersection = susp_lemmas_set & cand_lemmas_set
                    left_num = sum(map(intersection.__contains__, susp_lemmas))
                    right_num = sum(map(intersection.__contains__, cand_lemmas))
                    left_incl_dists.append(1 - left_num / len(susp_lemmas))
                    right_incl_dists.append(1 - right_num / len(cand_lemmas))
                    susp_lens_dists.append(len(susp_lemmas))
                    cand_lens_dists.append(len(cand_lemmas))
                    min_lens_dists.append(min(len(susp_lemmas), len(cand_lemmas)))
                top_dists.append(left_incl_dists)
                top_dists.append(right_incl_dists)
                top_dists.append(susp_lens_dists)
                top_dists.append(cand_lens_dists)
                top_dists.append(min_lens_dists)
                # Шаг 3: запись признаков
                for dists, ix in zip(zip(*top_dists), top_indices):
                    features.append(dists)
                    indices.append(ix)
            probas = clf.predict_proba(features)[:, 1]
            for (susp_sent_id, src_sent_id), p in zip(indices, probas):
                if p > 0.5:
                    _, susp_sent_i = sentences_files_ids[susp_sent_id]
                    _, src_sent_i = sentences_files_ids[src_sent_id]
                    pre_detections.append((p, susp_sent_i, src_sent_i))
            pre_detections_list.append(pre_detections)

threshold = 0.85

detections_list = []

for pre_detections in pre_detections_list:
    pre_detections = [(b, c) for a, b, c in pre_detections if a > threshold]
    detections_list.append(pre_detections)

logging.info("Read files lengths.")    

files_lens = {}

for susp_path, src_path in tqdm(test_paths):
    if susp_path not in files_lens:
        files_lens[susp_path] = sent_tokenize(open(data_dir + susp_path).read())[1]
    if src_path not in files_lens:
        files_lens[src_path] = sent_tokenize(open(data_dir + src_path).read())[1]

# Попытка уменьшить granularity 2.0.

logging.info("Decrease granularity.")

max_susp_gap, max_src_gap = 10, 10

out_detections_list = []
ge_by_snd = lambda x, xs: set(map(lambda p: int(p.split("_")[0]), filter(lambda p: p.endswith("_%d" % x), xs)))

for (susp_path, src_path), detections in tqdm(zip(test_paths, detections_list)):
    susp_lens = files_lens[susp_path]
    src_lens = files_lens[src_path]
    detections_set = set(detections)
    dsu_domain = set()
    out_detections = []
    for susp_sent_i, src_sent_i in detections_set:
        dsu_domain.add("%d_0" % susp_sent_i)
        dsu_domain.add("%d_1" % src_sent_i)
    if dsu_domain:
        dsu = UnionFind(list(dsu_domain))
        for susp_sent_i, src_sent_i in detections_set:
            dsu.union("%d_0" % susp_sent_i, "%d_1" % src_sent_i)
            for i in range(max_susp_gap + 1):
                for j in range(max_src_gap + 1):
                    if (susp_sent_i + i, src_sent_i + j) in detections_set:
                        dsu.union("%d_0" % susp_sent_i, "%d_0" % (susp_sent_i + i))
                        dsu.union("%d_0" % (susp_sent_i + i), "%d_1" % (src_sent_i + j))
        for comp in dsu.components():
            comp_susp = ge_by_snd(0, comp)
            comp_src = ge_by_snd(1, comp)
            susp_pos = (susp_lens[min(comp_susp)][0], susp_lens[max(comp_susp)][1] - 1)
            src_pos = (src_lens[min(comp_src)][0], src_lens[max(comp_src)][1] - 1)
            out_detections.append((src_pos, susp_pos))
    out_detections_list.append(out_detections)

logging.info("Write out detections.")

with open("detections.dump", "wb") as fout:
    pickle.dump(out_detections_list, fout)

logging.info("Done!")