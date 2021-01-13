# content-based recommendation model

This is an implementation of combining doc2vec and tf-idf word embedding methods with some binary feature using cross product similarity to generate a content-based model.

# output :
1. return pandas dataframe
    a. header : ​myvdo_content_id, NTU_REC_VD
    b. content : ​myvdo_content_id and its corresponding top N similar videos (type: str saperator: , )

# input :
1. ('video_metadata.csv', 'prd_off_myvdo_ttl.csv', 'video_metadata_top_cover_asoc.csv', additional_files,'stopwords.txt', 'word_set.txt', length)
2. additional_files = ['video_metadata_actor_cht.csv','video_metadata_drctr_cht.csv','video_metadata_gnr.csv','video_metadata_issr_cntry_nm.csv','video _metadata_scenarist.csv]
3. length must be 30-100 (default = 30)
