import pickle
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from hanziconv import HanziConv
import jieba

def tfidf(metadata_df, top_cover_dict, length):
    movie_list = metadata_df.content_desc.values
    movie_ids =  metadata_df.content_id.values

    transformer = TfidfVectorizer(max_features=10000)
    tfidf=transformer.fit_transform(movie_list)
    
    is_valid = metadata_df.is_valid.values
    weight=tfidf.toarray()
  
    np_weight = np.array(weight)

    similarities =  np.matmul(weight / np.linalg.norm(weight, axis=1)[:,None],\
                              (weight/ np.linalg.norm(weight, axis=1)[:,None]).T )

    # set similarity of  invalid videos as 0
    similarities[:,np.where(is_valid==0)[0]] = 0
    #sorting similarities
    sort_similarities = np.argsort(-similarities)

    sort_similarities =  np.array([ ss[ss!= i] for i, ss in enumerate(sort_similarities)])
    #out_np為最後輸出之100部相似電影item_id
    out_np=np.empty((len(movie_list),length),dtype='object')
    for i in range(0,len(sort_similarities)):
        idx = 0

        for vid in  sort_similarities[i]:
            recommend_video = top_cover_dict.get(movie_ids[vid], movie_ids[vid])
            if recommend_video not in out_np[i,:idx]:
                # if the video has not been recommended
                out_np[i,idx] = recommend_video
                idx += 1
                if idx == length:
                    break
    return np_weight, out_np


def getChinese(context):
#     filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
#     context = filtrate.sub(r'', context) # remove all non-Chinese characters
    context = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",context)
    context = re.sub("[【】╮╯▽╰╭★→「」-]+","",context)
    context = re.sub("[！，❤。～《》：（）【】「」？”“；：、]","",context)
    return context
    

# # THREE MODELS
# 
# * Doc2Vec + TF-IDF
# * Doc2Vec + Binary feature
# * TF-IDF

from collections import Counter
def merge(a_recs, b_recs, c_recs, length):
    '''
    return the merge of three recommendations
    '''
    # counter
    counter = Counter(a_recs.tolist() + b_recs.tolist() + c_recs.tolist())
    
    # all three recommender recommend are sorted top
    common = [ a_rec for a_rec in a_recs if counter[a_rec] == 3]
    
    result = []
    
    result += common
        
    appended = 0
    i=0
    while appended<= int((length+2)/3) and i < len(a_recs):
        if a_recs[i] not in result:
            result.append(a_recs[i])        
            appended += 1

        i += 1
    appended = 0
    i=0
    while appended<= int((length+2)/3) and i < len(b_recs):
        if b_recs[i] not in result:
            result.append(b_recs[i])   
            appended += 1
        i += 1
    appended = 0
    i=0
    while appended<= int((length+2)/3) and i < len(c_recs):
        if c_recs[i] not in result:
            result.append(c_recs[i])   
            appended += 1
        i += 1    
    
    return np.array(result[:length])
    
    
def remove_sepcail_segment(content, jieba_stop_words):
    content = HanziConv.toSimplified(content)
    seg_list = jieba.cut(content)
    seg_clean = []
    
    for word in seg_list:
        clean_word = getChinese(word).strip()
        if  clean_word== '':
            continue
        seg_clean.append(clean_word)
    
    seg_clean = [word for word in seg_clean if not word in jieba_stop_words]
    return ','.join(seg_clean)
    
    
def generate_VecDict(metadata_df):

    # create mapping between 
    mapping = {idx: ids for idx, ids in enumerate(metadata_df.myvdo_content_id.values)}
    documents = [TaggedDocument(doc.split(','), [i]) for i, doc in enumerate(metadata_df.myvdo_content_desc.values)]
        
    model = Doc2Vec(documents,vector_size=100,window=10, min_count=1, workers=16, alpha=0.008)
    model.train(documents,epochs=20,total_examples=model.corpus_count)

    vec_dict={}
    for idx, item_vec in enumerate(model.docvecs.vectors_docs):      
        vec_dict[mapping[idx]] = item_vec
         
    return vec_dict




def ntu_content_based_recommend(metadata_file, attribute_data_file, top_cover_file, additional_files, stopword_file='stopwords.txt', userdict_file='word_set.txt', length=30):
    
    # read stop words
    with open(stopword_file,'r') as f:
        jieba_stop_words = f.read().split('\n')
    
    # load self-defined dictionary
    jieba.load_userdict(userdict_file)
    
    # mapping from content id to top cover
    top_cover_df = pd.read_csv(top_cover_file).loc[:,['content_id','top_cover']]
    top_cover_df.loc[top_cover_df.top_cover.isna(),'top_cover'] =\
    top_cover_df.loc[top_cover_df.top_cover.isna(), 'content_id']
    top_cover_dict = { m:t for m, t in zip(top_cover_df.content_id, top_cover_df.top_cover )}


    metadata_df = pd.read_csv(metadata_file)    
    
    movie_list = metadata_df.content_desc.values
    movie_ids = metadata_df.content_id.values
    
    # preprocessing 
    additional_columns = []
    for f in additional_files:
        df  = pd.read_csv(f)
        current_additional_columns = [c for c in df.columns if c != 'content_id']
        additional_columns += current_additional_columns

        for col in current_additional_columns:            
            metadata_df = metadata_df.merge(df.groupby('content_id')[col].apply(list)\
                                            .apply(lambda x:' '.join(map(str,x))).reset_index(), on='content_id',how='left')
    for col in additional_columns + ['content_nm','kywrd']:
        metadata_df[col].fillna("",inplace=True)
        if col == 'ttl_gnr_id':
            metadata_df[col] = metadata_df[col].apply(lambda x:(x + " ")*3 )
        elif col == 'kywrd':
            metadata_df[col] = metadata_df[col].apply(lambda x:(x + " ")*2 )
        metadata_df['content_desc'] = metadata_df['content_desc'].str.cat(metadata_df[col].fillna("")).apply(lambda x:x+" ")
       
    metadata_df['content_desc'] = metadata_df['content_desc'].apply(lambda x:remove_sepcail_segment(x, jieba_stop_words))
    
    # TFIDF
    tfidf_embeddings, tfidf_recommend = tfidf(metadata_df, top_cover_dict, length)


    is_valid = metadata_df.is_valid.values

    # generate d2v embeddings  
    d2v_embedding_dict = generate_VecDict(metadata_df)
    
    # concatenate tfidf and d2v
    d2v_tfidf_embeddings = []
    for idx, d2v_embedding in enumerate(d2v_embedding_dict.values()):
        d2v_tfidf_embeddings.append(np.concatenate([d2v_embedding, tfidf_embeddings[idx]]))

    d2v_tfidf_embeddings = np.vstack(d2v_tfidf_embeddings)

    # compute item similarity based on d2v and tfidf
    d2v_tfidf_item2item_simiarity = np.matmul(d2v_tfidf_embeddings / np.linalg.norm(d2v_tfidf_embeddings, axis=1)[:,None], (d2v_tfidf_embeddings/ np.linalg.norm(d2v_tfidf_embeddings, axis=1)[:,None]).T )
    
    # filter out invalid movies and identity movie 
    d2v_tfidf_item2item_simiarity[:,np.where(is_valid==0)[0]] = 0
    d2v_tfidf_item2item_simiarity[np.arange(len(d2v_tfidf_item2item_simiarity)), np.arange(len(d2v_tfidf_item2item_simiarity))] = 0
    
    video_lists = metadata_df.content_id.values

    d2v_tfidf_order = np.array(list(map(lambda x: np.flip(np.argsort(x)), d2v_tfidf_item2item_simiarity)))

    d2v_tfidf_recommend = np.empty((len(movie_list),length),dtype='object')
    for i in range(0,len(d2v_tfidf_order)):
        idx = 0

        for vid in  d2v_tfidf_order[i]:
            recommend_video = top_cover_dict.get(movie_ids[vid], movie_ids[vid])
            if recommend_video not in d2v_tfidf_recommend[i,:idx]:
                # if the video has not been recommended
                d2v_tfidf_recommend[i,idx] = recommend_video
                idx += 1
                if idx == length:
                    break
    
    
    # doc2vec + binary
    # read attribute data
    attribute_df= pd.read_csv(attribute_data_file)
    metadata_df = metadata_df.merge(attribute_df.loc[:,['CONTENT_ID','MAIN_TTL_GNR_ID_SYS','ISSR_CNTRY_NM','LANG']].rename(columns={'CONTENT_ID': 'content_id'}), on='content_id', how='left')
    
    # create one hot array
    binary_df = pd.get_dummies(metadata_df.loc[:,['MAIN_TTL_GNR_ID_SYS','ISSR_CNTRY_NM','LANG']])
    
    # concatenate d2v and binary
    d2v_binary_embeddings = []
    for idx, d2v_embedding in enumerate(d2v_embedding_dict.values()):
            d2v_binary_embeddings.append(np.concatenate([d2v_embedding, binary_df.values[idx]]))

        
    d2v_binary_embeddings = np.vstack(d2v_binary_embeddings)
    
    # compute item similarity based on d2v and tfidf
    d2v_binary_item2item_simiarity = np.matmul(d2v_binary_embeddings / np.linalg.norm(d2v_binary_embeddings, axis=1)[:,None], (d2v_binary_embeddings/ np.linalg.norm(d2v_binary_embeddings, axis=1)[:,None]).T )
    
    # filter out invalid movies and identity movie 
    d2v_binary_item2item_simiarity[:,np.where(is_valid==0)[0]] = 0
    d2v_binary_item2item_simiarity[np.arange(len(d2v_binary_item2item_simiarity)), np.arange(len(d2v_binary_item2item_simiarity))] = 0
    
    d2v_binary_order = np.array(list(map(lambda x: np.flip(np.argsort(x)), d2v_binary_item2item_simiarity)))
    
    d2v_binary_recommend = np.empty((len(movie_list),length),dtype='object')
    for i in range(0,len(d2v_binary_order)):
        idx = 0

        for vid in  d2v_tfidf_order[i]:
            recommend_video = top_cover_dict.get(movie_ids[vid], movie_ids[vid])
            if recommend_video not in d2v_binary_recommend[i,:idx]:
                # if the video has not been recommended
                d2v_binary_recommend[i,idx] = recommend_video
                idx += 1
                if idx == length:
                    break
    
    final_recommendation = np.array([ merge(td ,dt, db, length) for td ,dt, db in zip(tfidf_recommend, d2v_tfidf_recommend, d2v_binary_recommend)])
    
    result = pd.DataFrame()
    result['myvdo_content_id'] = metadata_df.myvdo_content_id
    result['NTU_REC_VD'] = [ ','.join(rec) for rec in final_recommendation]
    
    # store to local
    result.to_csv('NTU_recommendations.csv',index=None)
    
    return result


