# IM_smoothed_greedy
 
## Datasets : 

Weibo dataset :
    
    https://www.lri.fr/owncloud/index.php/s/6jnbYJjnZS9Q8SL?path=%2FSina%20Weibo%20Dataset

Twitter dataset :

    https://www.dropbox.com/sh/g13q1yxpahtujvp/AABv48Av_flaemZXLlnV2Nnea?dl=0

Directories : 

    data/
        weibo/weibodata/ : raw data
        weibo_preprocessed/ : data from preprocessing_weibo.py
        weibo_features/ : data from features_engineering_weibo.py
        instances_weibo/ : data from create_instances.py
        twitter/ : raw data
        twitter_preprocessed/ : data from preprocessing_twitter.py
    
    results/
        results returned by grd_main.py, results.py

## Files : 

### preprocessing_weibo.py : 

Converts all the .txt files of data/weibo directory into .pkl files.
The subsampling of the graph can be controlled by the parameters.

**Execution :** 

    dense subsampling :
    ```
    python preprocessing_weibo.py --subsampling 2 --n1 -1 --n2 150
    ```
    sparse subsampling :
    ```
    python preprocessing_weibo.py --subsampling 1 --n1 5000 --n2 10
    ```

input :

    Parameters : 
    subsampling :
        1 for selecting the first n2 reposts of the first n1 cascades
        2 for selecting the reposts of targets having more than n2 reposts among the n1 first cascades
    n1, n2

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

output :

    userProfile.pkl : 
        i:uid(int64) - bi_followers_count(int32) - city(cat) - verified(cat) - followers_count(int32) - location(object) - province(category) - friends_count(int32) - name(object) - gender(category) -  created_at(object) - verified_type(category) -  statuses_count(int32) - description(object)
        shape : (1681085, 14)
    infos_influencers3.pkl : 
        i:userid(int64) - total_likes - total_reposts - n_cascades - 0 - 1 - 2 
        shape : (47555, 6)
    infos_targets3.pkl : 
        i:userid(int64) - 0 - 1 - 2
        shape : (1334887, 3)
    user_cascades.pkl : 
        i:v(int32) - mids(list(int64)) - Av(int32)
    infos_cascades.pkl : 
        i:n_cascades - mid(int64) - date(pd.DateTime) - u(int32) - n_likes(int32) - n_reposts(int32) - users2(list(int32))
    labels{subsampling}_{n1}_{n2}.pkl : 
        i:index - u(int64) - v(int64) - BT(float) - JI(float) - LP(float)
    edges{subsampling}_{n1}_{n2}.pkl :  
        i:index - u(int64) - v(int64)
    influencers_infector.pkl 
        shape : (1170688, 50)
    targets_infector.pkl
        shape : (1170688, 50)

### feature_engineering_weibo.py : 

Computes the features of the subsampled graph given by preprocessing_weibo.py. 0,1,2 are the topic classification features

**Execution :** 

    ```
    python feature_engineering_weibo.py --labels-name 'labels2_-1_150.pkl' --edges-name 'edges2_-1_150.pkl' --features-influencers-name 'features_influencers2_-1_150.pkl' --features-targets-name 'features_targets2_-1_150.pkl'
    ```

input : 

    userProfile.pkl
    --labels-name : labels{subsampling}_{n1}_{n2}.pkl
    --edges-name : edges{subsampling}_{n1}_{n2}.pkl
    infos_influencers3.pkl
    infos_targets3.pkl

output : 

    --features-influencers-name : features_influencers{subsampling}_{n1}_{n2}.pkl : 
        i:uid(int64) - followers_count(float64) - friends_count(float64) - statuses_count(float64) - verified(int8) - gender(int8) - d_out(float64) - pagerank(float64) - total_likes(float64) - total_reposts(float64) - n_cascades(float64) - 0(float64) - 1(float64) - 2(float64)
    --features-targets-name : features_targets{subsampling}_{n1}_{n2}.pkl : 
        i:uid(int64) - followers_count(float64) - friends_count(float64) - statuses_count(float64) - verified(int8) - gender(int8) - d_in(float64) - pagerank(float64) - 0(float64) - 1(float64) - 2(float64)


### create_instances.py

Given the features, labels, and edges, this file creates instances to feed the model

**Execution :** 
Modify the parameters in the file and execute
    
    ```
    python create_instances.py
    ```

input : 

    Parameters :
    output_dir : output folder         
    N_INSTANCES : number of instances to create
    N_INFLUENCERS, N_TARGETS : size of instances
    PROP_I : proportion of best influencers considered
    PROB_TYPE (BT, JI, LP) : Method for computing the data-based labels

    features_influencers : returned by feature_engineering_weibo.py
    features_targets.pkl : returned by feature_engineering_weibo.py
    labels.pkl : returned by preprocessing_weibo.py
    edges.pkl :  returned by preprocessing_weibo.py
    influencers_embeddings :  returned by preprocessing_weibo.py
    targets_embeddings :  returned by preprocessing_weibo.py
    
output : 

    N_INSTANCES instances in the 'output_dir' directory, of shape (N_INFLUENCERS, N_TARGETS, N_FEATURES + 3)

### greedy_coverage_gpu.py

    Contains functions for the decision-focused learning.

### greedy_submodular_new.py

    Contains the class of the GreedyOptimizer and the StochasticGreedyOptimizer

### grd_main.py 

Training of the 2-stage and the decision focused model
    
**Execution :** 
Modify the instance path, N_INSTANCE, N_INFLUENCERS, N_TARGETS, N_FEATURES in the file and execute 
    
    ```
    python grd_main.py --n-iter 15 --output_dir 'results/experience1/' --device 'cpu'
    ```

input : 

    --net-df/2s-path : name of the models
    --labels : db, infector or inf2vec
    --n-iter : number of different models trained
    --output-dir : path where the results are generated
    --device : device on which the file runs. 'cpu', 'cuda:0', 'cuda:1'...
    
    instance_path : folder containing the N_INSTANCES of shape (N_INFLUENCERS, N_TARGETS, N_FEATURES + labels)
    N_INSTANCE, N_INFLUENCERS, N_TARGETS, N_FEATURES


output (in output_dir) : 

    Models :
    output_dir/net_df_path_{labels}_{i}.pt for i in range(n_iter) : the n_iter df models trained
    output_dir/net_2s_path_{labels}_{i}.pt for i in range(n_iter) : the n_iter 2s models trained

    output_dir/df_training.txt : logs of the training of the df models
        epoch | loss | train_score | test_score | dni_train | dni_test | mean(pred) | learning_rate
        0 | -25.949039459228516 | 159.42720127105713 | 162.54154205322266 | 326.25 | 334.75 | 0.018852414563298225 | 0.0007750000000000001
        ...
    
    output_dir/2s_training.txt : same for 2s models

    output_dir/perfs_train_test.txt : average final performances for each of the n_iter models for various values of K (number of seeds)
    (Exp-train df, Exp-test df, DNI-train df, DNI-test df, Exp-train 2s, Exp-test 2s, DNI-train 2s, DNI-test 2s, Exp-train rnd, Exp-test rnd, DNI-train rnd, DNI-test rnd, Exp-train grd, Exp-test grd, DNI-train grd, DNI-test grd, Exp-train deg, Exp-test deg, DNI-train deg, DNI-test deg)
        Example : 
        Exp-train df, 112.97524213790894,172.15905284881592,276.28208351135254,357.4610347747803,418.4751853942871
        ...

    output_dir.baselines.txt : average performances of random algorithm, greedy-oracle algorithm and degree heuristic for each of the n_iter models. (They are varying because the train/test split is different for every model)

### results.py 

The models returned by grd_main.py can be tested on other instances : sparse instances, twitter instances.

**Execution :**
    
    ```
    python results.py --models-path 'results/experience/' --n-iter 15 --instances-path 'data/weibo_instances/sparse/' --n-instances 20 --labels db --file_name 'perfs_sparse.txt'
    ```

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


### convert_result_file.py 

Converts the results of output_dir/perfs_train_test.txt or the output of results.py in a .csv file containing the avg and the std of the n_iter different models

**Execution : **

    ```
    python convert_results_file.py --n-iter 15 --file-path 'results/experience1/perfs_sparse.txt' --sparse True
    python convert_results_file.py --n-iter 15 --file-path 'results/experience1/perfs_train_test.txt' 
    ```

input : 

    --n-iter : number of models
    --file-path : file to convert
    --sparse : to keep empty if the file is perfs_train_test.txt, to put to 'True' if the file is from results.py. (it changes the number of columns)

output : 

    {file-path}_mean.csv : contains the means of the performances of the models
    {file-path}_std.csv : contains the standard deviations of the performances of the models

### vizualize_training.py

**Execution : **

    ```
    python vizualize_training.py --input-dir 'results/experience1/df_training.txt' --title 'Training of the decision-focused models of Experience 1 '
    ```

input : 

    output_dir/df_training.txt or 2s_training.txt

output : 

    plots graphs of the different values wrt the epochs

## References

Main paper : 
Shinsaku Sakaue, Differentiable Greedy Submodular Maximization: Guarantees, Gradient Estimators, and Applications. [DOI](
https://doi.org/10.48550/arXiv.2005.02578)

Fan Zhou, Xovee Xu, Goce Trajcevski, and Kunpeng Zhang. 2021. A Survey of Information Cascade Analysis: Models, Predictions, and Recent Advances. ACM Comput. Surv. 54, 2, Article 27 (March 2022), 36 pages. [DOI](https://doi.org/10.1145/3433000)

Amit Goyal, Francesco Bonchi, and Laks V.S. Lakshmanan. 2010. Learning influence probabilities in social networks. In Proceedings of the third ACM international conference on Web search and data mining (WSDM '10). Association for Computing Machinery, New York, NY, USA, 241–250. [DOI](https://doi.org/10.1145/1718487.1718518)

G. Panagopoulos, F. Malliaros and M. Vazirgiannis, "Multi-task Learning for Influence Estimation and Maximization," in IEEE Transactions on Knowledge and Data Engineering, [DOI] (https://doi.org/10.1109/TKDE.2020.3040028)

Ko, K. Lee, K. Shin and N. Park, "MONSTOR: An Inductive Approach for Estimating and Maximizing Influence over Unseen Networks," in 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), The Hague, Netherlands, 2020 pp. 204-211.
[DOI] (https://doi.org/10.1109/ASONAM49781.2020.9381460)


