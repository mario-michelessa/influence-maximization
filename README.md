# IM_smoothed_greedy
 
Files : 

preprocessing_weibo.py : 
    input :
        User profile information : 
        data/weibo/weibodata/userProfile/userprofile1.txt
        data/weibo/weibodata/userProfile/userprofile2.txt
        
        Index information : 
        data/weibo/weibodata/diffusion/uidlist.txt
        data/weibo/weibodata/diffusion/repostidlist.txt
        
        Topic information :
        data/weibo/weibodata/topic-100/doc.txt
        
        Cascade Information : 
        data/weibo/weibodata/total.txt

        Social graph information : 
        data/weibo/weiodata/graph_170w_1month.txt

        Embeddings : 
        data\weibo\weibo_embedding\mtl_n_target_embeddings_p.txt
        data\weibo\weibo_embedding\mtl_n_source_embeddings_p.txt

        Parameters : 
            subsampling :
                1 for selecting the first n2 reposts of the first n1 cascades
                2 for selecting the reposts of targets having more than n2 reposts among the n1 first cascades
            n1, n2

    output :
        userProfile.pkl : i:uid(int64) - bi_followers_count(int32) - city(cat) - verified(cat) - followers_count(int32) - location(object) - province(category) - friends_count(int32) - name(object) - gender(category) -  created_at(object) - verified_type(category) -  statuses_count(int32) - description(object)
            shape : (1681085, 14)
        infos_influencers3.pkl : i:userid(int64) - total_likes - total_reposts - n_cascades - 0 - 1 - 2 
            shape : (47555, 6)
        infos_targets3.pkl : i:userid(int64) - 0 - 1 - 2
            shape : (1334887, 3)
        user_cascades.pkl : i:v(int32) - mids(list(int64)) - Av(int32)
        infos_cascades.pkl : i:n_cascades - mid(int64) - date(pd.DateTime) - u(int32) - n_likes(int32) - n_reposts(int32) - users2(list(int32))
        labels{subsampling}_{n1}_{n2}.pkl : i:index - u(int64) - v(int64) - BT(float) - JI(float) - LP(float)
        edges{subsampling}_{n1}_{n2}.pkl :  i:index - u(int64) - v(int64)
        influencers_infector.pkl 
            shape : (1170688, 50)
        targets_infector.pkl
            shape : (1170688, 50)

feature_engineering_weibo.py : 
    
    input : 
        userProfile.pkl
        labels
        edges
        infos_influencers
        infos_targets

    output : 
        features_influencers{subsampling}_{n1}_{n2}.pkl - 
        features_targets{subsampling}_{n1}_{n2}.pkl
