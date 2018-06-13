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

#from tqdm import tqdm
from unionfind.unionfind import UnionFind
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm = lambda x: x

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

# TODO
data_dir = sys.argv[1] + "/"
tasks_dir = data_dir

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

with open("classifier2.0_dialog_all_random1.mdl", "rb") as fin:
    clf = pickle.load(fin)


# Create dataset out of embeddings ranking.

pre_detections_list = []

test_paths = []

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
            features, indices, pre_detections = [], [], []
            sents_susp_dict, sents_src_dict = sentences_dict[susp_path], sentences_dict[src_path]
            embed_dists = []
            for embeddings in embeddings_list:
                susp_embeddings = embeddings[susp_path]
                src_embeddings = embeddings[src_path]
                susp_vecs = list(map(lambda x: get_embedding(x, susp_embeddings), susp_sents_ids))
                cand_vecs = list(map(lambda x: get_embedding(x, src_embeddings), src_sents_ids))
                dists = 0.5 * (1 - cosine_similarity(susp_vecs, cand_vecs))
                embed_dists.append(dists)
            for t1, (susp_sent_i, susp_sent_id) in enumerate(zip(susp_sents_is, susp_sents_ids)):
                susp_lemmas = list(sents_susp_dict[susp_sent_i][1])
                susp_lemmas_set = set(susp_lemmas)
                # Step 1: calculate distance using all embeddings.
                top_indices = list(map(lambda x: (susp_sent_id, x), src_sents_ids))
                top_dists = []
                for dists in embed_dists:
                    top_dists.append(dists[t1])
                left_incl_dists = [] # susp_lemmas \in cand_lemmas
                right_incl_dists = [] # cand_lemmas \in susp_lemmas
                iou_incl_dists = [] # intersection \in union
                for cand_sent_id in src_sents_ids:
                    cand_path, cand_sent_i = sentences_files_ids[cand_sent_id]
                    assert cand_path == src_path
                    cand_lemmas = list(sents_src_dict[cand_sent_i][1])
                    cand_lemmas_set = set(cand_lemmas)
                    intersection = susp_lemmas_set & cand_lemmas_set
                    union = susp_lemmas + cand_lemmas
                    left_num = sum(map(intersection.__contains__, susp_lemmas))
                    right_num = sum(map(intersection.__contains__, cand_lemmas))
                    iou_num = sum(map(intersection.__contains__, union))
                    left_incl_dists.append(1 - left_num / len(susp_lemmas))
                    right_incl_dists.append(1 - right_num / len(cand_lemmas))
                    iou_incl_dists.append(1 - iou_num / len(union))
                top_dists.append(left_incl_dists)
                top_dists.append(right_incl_dists)
                top_dists.append(iou_incl_dists)
                # Step 2: add all dists to features.
                for dists, ix in zip(zip(*top_dists), top_indices):
                    features.append(dists)
                    indices.append(ix)
            features = np.array(features)
            assert features.shape[1] == 6
            features_2d = np.vstack([np.mean(features[:, :3], axis=1), np.mean(features[:, 3:6], axis=1)]).T
            probas = clf.predict_proba(features_2d)[:, 1]
            #probas = kde_decision_function(features_2d)
            for (susp_sent_id, src_sent_id), p in zip(indices, probas):
                #if p > 0.25:
                if p > 0.9999:
                    _, susp_sent_i = sentences_files_ids[susp_sent_id]
                    _, src_sent_i = sentences_files_ids[src_sent_id]
                    pre_detections.append((p, susp_sent_i, src_sent_i))
            pre_detections_list.append(pre_detections)

threshold = 0.99998

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

# Granularity reduction.

logging.info("Decrease granularity.")

max_susp_gap, max_src_gap = 10, 10

out_detections_list = []

for (susp_path, src_path), detections in tqdm(zip(test_paths, detections_list)):
    susp_lens = files_lens[susp_path]
    src_lens = files_lens[src_path]
    detections = sorted(detections)
    detections_set = set(detections)
    out_detections = set()
    edge_dsu = {}
    edge_dsu_cnt = 0
    for susp_sent_i, src_sent_i in detections:
        if (susp_sent_i, src_sent_i) not in edge_dsu:
            edge_dsu_cnt += 1
            edge_dsu[susp_sent_i, src_sent_i] = edge_dsu_cnt
        edge_comp = edge_dsu[susp_sent_i, src_sent_i]
        for i in range(max_susp_gap + 1):
            for j in range(max_src_gap + 1):
                if (susp_sent_i + i, src_sent_i + j) in detections_set:
                    edge_dsu[susp_sent_i + i, src_sent_i + j] = edge_comp
    dsu_edge = {}
    for (u, v), n in edge_dsu.items():
        if n not in dsu_edge:
            dsu_edge[n] = (set(), set())
        dsu_edge[n][0].add(u)
        dsu_edge[n][1].add(v)
    for susp_sent_i, src_sent_i in detections:
        edge_comp = dsu_edge[edge_dsu[susp_sent_i, src_sent_i]]
        comp_susp, comp_src = edge_comp
        susp_pos = (susp_lens[min(comp_susp)][0], susp_lens[max(comp_susp)][1])
        src_pos = (src_lens[min(comp_src)][0], src_lens[max(comp_src)][1])
        out_detections.add((src_pos, susp_pos))
    out_detections_list.append(list(out_detections))

logging.info("Write out detections.")

with open("detections.dump", "wb") as fout:
    pickle.dump(out_detections_list, fout)

logging.info("Done!")
