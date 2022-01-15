import numpy as np
import pandas as pd
import string
import texthero as hero
from nltk.util import ngrams
from fasttext import load_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import Pipeline

import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency

import torch
import transformers

from transformers import BertTokenizer
from util import AbstractBaseBlock


class StringLengthBlock(AbstractBaseBlock):
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].str.len()
        return out_df.add_prefix('StringLength_')


class GetCountWord(AbstractBaseBlock):
    """単語数を取得するブロック"""

    def __init__(self, cols):
        self.cols = cols

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.cols] = cleansing_hero_text(input_df[self.cols])
        out_df[self.cols] = out_df[self.cols].apply(
            lambda x: len(x.split())).fillna("")
        return out_df.add_prefix("count_word_")


def cleansing_hero_text(text_col):
    """
    Stopwords抜きの関数
    """
  custom_pipeline = [
                   hero.preprocessing.remove_whitespace,
                   hero.preprocessing.drop_no_content,
                   hero.preprocessing.fillna,
                   hero.preprocessing.remove_diacritics,
                   hero.preprocessing.remove_digits,
                   hero.preprocessing.remove_html_tags,
                   hero.preprocessing.remove_urls,
                   hero.preprocessing.remove_brackets,
                   hero.preprocessing.remove_punctuation,
                   hero.preprocessing.stem]
  texts = hero.clean(text_col, custom_pipeline)
  return texts

def cleansing_hero_text_and_stopwords(text_col):
    
    """
    stopwordsを含む
    """
  custom_pipeline = [
                   hero.preprocessing.remove_whitespace,
                   hero.preprocessing.drop_no_content,
                   hero.preprocessing.fillna,
                   hero.preprocessing.remove_diacritics,
                   hero.preprocessing.remove_digits,
                   hero.preprocessing.remove_html_tags,
                   hero.preprocessing.remove_urls,
                   hero.preprocessing.remove_brackets,
                   hero.preprocessing.remove_punctuation,
                   hero.preprocessing.stem,
                   hero.preprocessing.remove_stopwords]
  texts = hero.clean(text_col, custom_pipeline)
  return texts

def line_ngram(line, n=2):
  words = [w for w in line.split(' ') if len(w) != 0]
  return list(ngrams(words, n))

def create_n_gram(x, n=3):
  x = cleansing_hero_text(x)
  x = pd.Series(x).map(lambda r: line_ngram(r, n=n))
  return x

class NameNGramBlock(AbstractBaseBlock):
    def __init__(self, whole_df, col, n=3):
        self.whole_df = whole_df
        self.col = col
        self.n = n

    def fit(self, input_df, y=None):
        name_grams = create_n_gram(self.whole_df[self.col], n=self.n)
        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(20).index

        self.top_grams_ = top_grams
        return self.transform(input_df)

    def transform(self, input_df):
        name_grams = create_n_gram(input_df[self.col], n=self.n)
        output_df = pd.DataFrame()

        for top in self.top_grams_:
            s_top = '-'.join(top)
            output_df[f'{s_top}'] = name_grams.map(lambda x: top in x).map(int)

        return output_df.add_prefix('Name_has_').add_suffix(f'_n={self.n}')

def text_normalization(text):

    # 英語とオランダ語を stopword として指定
    # 複数の言語を処理したい場合は逐一変更
    custom_stopwords = nltk.corpus.stopwords.words('english')

    x = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])

    return x
      
class TfidfBlock(AbstractBaseBlock):
    """tfidf x SVD による圧縮を行なう block"""
    def __init__(self, column: str):
        """
        args:
            column: str
                変換対象のカラム名
        """
        self.column = column

    def preprocess(self, input_df):
        x = text_normalization(input_df[self.column])
        return x

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す. 
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = self.preprocess(input_df)
        self.pileline_ = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('svd', TruncatedSVD(n_components=50)),
        ])

        self.pileline_.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pileline_.transform(text)

        out_df = pd.DataFrame(z)
        
# reference: https://github.com/arosh/BM25Transformer/blob/master/bm25.py
class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        return X
    
class BertSequenceVectorizer:
    def __init__(self, model_name="bert-base-uncased", max_len=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_len

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()

    
class GetLanguageLabel(AbstractBaseBlock):
    """
    言語判定するブロック
    """
    def __init__(self, cols, path):
      self.cols = cols
      self.path = path
    def fit(self, input_df):
      self.model = load_model(self.path)
      return self.transform(input_df)

    def transform(self, input_df):
      out_df = pd.DataFrame()
      out_df[self.cols] = input_df[self.cols].fillna("").map(lambda x: self.model.predict(x.replace("\n", ""))[0][0])
      return out_df.add_prefix("lang_label_")
