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
        labels{subsampling}_{n1}_{n2}.pkl
        edges{subsampling}_{n1}_{n2}.pkl
        infos_influencers3.pkl
        infos_targets3.pkl

    output : 
        features_influencers{subsampling}_{n1}_{n2}.pkl - 
        features_targets{subsampling}_{n1}_{n2}.pkl

    Computes the features of the subsampled graph given by preprocessing_weibo.py. 0,1,2 are the topic classification features
    features_influencers : i:uid(int64) - followers_count(float64) - friends_count(float64) - statuses_count(float64) - verified(int8) - gender(int8) - d_out(float64) - pagerank(float64) - total_likes(float64) - total_reposts(float64) - n_cascades(float64) - 0(float64) - 1(float64) - 2(float64)
    features_targets : i:uid(int64) - followers_count(float64) - friends_count(float64) - statuses_count(float64) - verified(int8) - gender(int8) - d_in(float64) - pagerank(float64) - 0(float64) - 1(float64) - 2(float64)

create_instances.py

    input : 
    
        features_influencers.pkl
        features_targets.pkl
        labels.pkl
        edges.pkl
        influencers_embeddings
        targets_embeddings
        
        parameters : 
            PROP_I : proportion of best influencers considered
            PROB_TYPE (BT, JI, LP) : Method for computing the data-based labels

    output : 
        N_INSTANCES instances in the 'path' directory, of shape (N_INFLUENCERS, N_TARGETS, N_FEATURES)

In the decision-focused-learning-gpu folder : 

greedy_coverage_gpu.py

    Contains functions for the decision-focused learning.

greedy_submodular_new.py

    Contains the class of the GreedyOptimizer and the StochasticGreedyOptimizer

grd_main.py 

    Training of the 2-stage and the decision focused model
    
    Parameters : 
        --net-df/2s-path : name of the models
        --labels : db, infector or inf2vec
        --n-iter : number of different models trained
        --output-dir : path where the results are generated
        --device : device on which the file runs. 'cpu', 'cuda:0', 'cuda:1'...
        
        instance_path : folder containing the N_INSTANCES of shape (N_INFLUENCERS, N_TARGETS, N_FEATURES + labels)

        output (in output_dir) : 
            output_dir/net_df_path_{labels}_{i}.pt for i in range(n_iter) : the n_iter df models trained
            output_dir/net_2s_path_{labels}_{i}.pt for i in range(n_iter) : the n_iter 2s models trained

            output_dir/df_training.txt : logs of the training of the df models
                n_iter - date
                epoch | loss | train_score | test_score | dni_train | dni_test | mean(pred) | learning_rate
                0 | -25.949039459228516 | 159.42720127105713 | 162.54154205322266 | 326.25 | 334.75 | 0.018852414563298225 | 0.0007750000000000001
                ...
            
            output_dir/2s_training.txt : same for 2s models

            output_dir/perfs_train_test.txt : average final performances for each of the n_iter models for various values of K (number of seeds)
            (Exp-train df, Exp-test df, DNI-train df, DNI-test df, Exp-train 2s, Exp-test 2s, DNI-train 2s, DNI-test 2s, Exp-train rnd, Exp-test rnd, DNI-train rnd, DNI-test rnd, Exp-train grd, Exp-test grd, DNI-train grd, DNI-test grd, Exp-train deg, Exp-test deg, DNI-train deg, DNI-test deg)
                Example : 
                {date} - Dataset : {instance_path} - labels : db - numepochs : 20 - batchsize : 5 - lr : 0.001 - regcoeff : 0.1 
                Ks : [5, 10, 25, 50, 100] 
                Exp-train df, 112.97524213790894,172.15905284881592,276.28208351135254,357.4610347747803,418.4751853942871
                ...

            output_dir.baselines.txt : average performances of random algorithm, greedy-oracle algorithm and degree heuristic for each of the n_iter models. (They are varying because the train/test split is different for every model)

results.py 

    The models returned by grd_main.py can be tested on other instances : sparse instances, twitter instances.

    input :
        --models-path : output_dir of the grd_main.py, containing the n_iter models
        --n-iter : number of models in the directory
        --instances-path : dataset tested : sparse or twitter instances for example
        --n-instances : number of instances in the test dataset
        --exp : if 'noBox' then we skip the transform_Y step, if noCas then we remove the cascade features
        --labels : db, infector or inf2vec
        --device : cpu or cuda:i
        --file-name : name of the output file : it will be saved in output_dir

    output : 
        output_dir/file-name.txt : 
            contains the performances of the n_iter models for different values of K : 
            (Exp df, DNI df, Exp 2s, DNI 2s, Exp rnd, DNI rnd, Exp grd, DNI grd, Exp deg, DNI deg)
            Example : 
            Exp df, 0.3021872669458389,0.587222883105278,1.385655701160431,2.744034457206726,5.421094512939453
            DNI df, 36.75,56.7,83.1,117.05,174.2
            ...
            
convert_result_file.py 

    Converts the results of output_dir/perfs_train_test.txt or the output of results.py in a .csv file containing the avg and the std of the n_iter different models

    input : 
        --n-iter : number of models
        --file-path : file to convert
        --sparse : to keep empty if the file is perfs_train_test.txt, to put to 'True' if the file is from results.py. (it changes the number of columns)

    output : 
        {file-path}_mean.csv : contains the means of the performances of the models
        {file-path}_std.csv : contains the standard deviations of the performances of the models

vizualize_training.py

    input : 
        output_dir/df_training.txt or 2s_training.txt
    
    output : 
        plots graphs of the different values wrt the epochs


