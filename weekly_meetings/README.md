# Shinkansen's Weekly Meeting Notes

[Overleaf report](https://www.overleaf.com/project/66fa55cd26323c2d085403fd)
#### Meeting Outline
* [03 October 2024](#date-03-october-2024)
* [10 October 2024](#date-10-october-2024)
* [05 November 2024](#date-05-november-2024)
* [14 November 2024](#date-14-november-2024)

#### Date: 03 October 2024

##### Who did you help this week?

*  No one

##### Who helped you this week?

* Youtube videos on state space models
* Meeting with Veronika last week

##### What did you achieve?

* Lost fear of State Space Models

##### What did you struggle with?

* State Space Models
* Defining a scope for our litterature review: How strict/loose?
* Pick and Choose correct data sets for benchmarking

##### What would you like to work on next week?

*  Compile dataset
*  Litterature Review
*  Model Testing/data sandbox
*  HPC

##### Where do you need help from Veronika?

* Litterature Review

#### Date: 10 October 2024


##### Who helped you this week?

* Novonesis on datasets
* Veronika

##### What did you achieve?

* Literature review nearly done.
    * Needs to be written in report.
* Dataset review is in progress.
    * Localized API endpoints and strategy to retrieve.

##### What did you struggle with?

* Optimal storage/access to data when processsing.  
* Eligibility criteria for literature review.
* OpenAlex API search query was giving funny results.

##### What would you like to work on next week?

*  Writing report for systematic literature review.
*  Datasets.

##### What would like to work on today?

* Familiarise ourselves the the metagenomic binning methods / tools / evaluation used in VAMB and DNABERT-S
    * Start Email correspondance with novonesis to verify new datasets and their eligibility - both for metagenomic binning and phenotype prediction.
* Write systematic literature review methodology in overleaf.
    * Start on introduction?

##### Where do you need help from Veronika?

* Processing large files on HPC and optimal processing.
* Discard systematic dataset review?
* PRISMA guidelines

#### Date: 05 November 2024


##### Who helped you this week?


##### What did you achieve (since last time)?

*  Literature review is finished.
    * Found 17 foundation models and 3 benchmark papers
*  We chose to pursue the metagenomics binning task, as phenotype prediction (discussed last time) was out of scope.
    * We found a dataset used in one of the key papers (VAMB) called MetaHIT.
    * Instead of raw reads as seen last time, this dataset consists of assembled raw reads into so called contigs (contiguous dna sequences)
        * All contigs are labeled with a species or genus id.
        * Contig lengths vary from 2500 to around 20000
        * The task of metagenomic binning in this project is now to represent each contig with embeddings from model *X* and apply the K-medoid unsupervised clustering algorithm described in [DNABERT-S](https://arxiv.org/pdf/2402.08777). Contigs and clusters are then labeled and compared to the true labels.
* Written source code for the whole metagenomics binning pipeline.
    * We managed to get embeddings from the two simplest models.

##### What did you struggle with?

* We struggled with the queues on the HPC cluster to get embeddings from huggingface models. We are now trying the external ressources Ucloud.
* We expect to use at least 24GB but probably more, as the K-medoid algorithm also uses some memory.

##### What would you like to work on today and next week?

* Get all embeddings. So getting the actual computation on the cluster to work.
* Quality check of linear sum alignment algorithm used by DNABERT-S to align predicted labels with the true labels.
    * This can be done locally using the already computed embeddings.

##### Where do you need help from Veronika?

* Decide on method for calculating threshold for the K-medoid clustering algorithm.
    * The K-medoid algorithm is sensitive to changes in the threshold paramater, that governs whether two contigs should be clustered together or not. In DNABERT-S they use a seperate dataset to calculate an independent threshold for each of the models, as they claim the magnitudes of distances vary a lot from model to model. The threshold is calculated using the seperate dataset by computing all distances from each contig to its respective species centroid and pick the 70th percentile out of all of these distances (a bit arbitrary). 
* Our method: 
    * We do not have such a hold out dataset but emulate this by sampling a certain number of contigs from all the species in the data to calculate the 70th percentile of the distances and use this as our threshold for each of our models.
    * Question 1: 
        * Should we sample x amount of contigs from each species to assure we have all the clusters when calculating the threshold?
        * Or should we sample all contigs within a percentage of the species, say 20% of the species?
    * Question 2:
        * The K-medoid algorithm is very sensitive to the threshold paramater, so the sampling could result in different threshold parameters by chanche. To avoid the randomness we could use the whole dataset to calculate the threshold. Would this be considered p-hacking (lack of better word)?
    * Question 3:
        * What are the pros and cons of using a threshold for each the seperate models instead of having a pre-set global threshold.
        * All the embeddings are normalized, so this threshold calculation seems a bit suspicious.
* Question 4:
    In DNABERT-S they discard all contigs that were not assigned any label from the K-medoid algorithm when calculating their results. We think that they do this to make the linear sum alignment algorithm work, as it assumes all contigs are given a label. 
    This is a clear downside where they discard unclassified contigs instead of reporting it. We thought to include the unclassified contigs in some way. Perhaps by randomly assigning them label, or using them in the results somehow.

    * What are your thoughts on this?


We would like to get feedback on our report and report structure so far.



#### Date: 14 November 2024

**Note that the latest code is found in** [branch andreas on our github](https://github.com/eisuke119/Research-Project/tree/andreas). 

##### Who helped you this week?

  * Lottie Greenwood was very helpful in helping us using Ucloud.

##### What did you achieve (since last time)?

  * Access to compute power on Ucloud. Here there is no queue compared to the cluster.
  * Code working for 6 DNA foundation models (LLMs) and 3 non-LLMs.  
  * The NxN similarity matrix in the k-mediod algoritm caused out-of-memory-errors. We made a new memory-efficient implementation.
  * Set aside 10% of species for calculating threshold. 



##### What did you struggle with?

  * Debugging code, out-of-memory-erorrs



##### What would you like to work on today and next week?

   1. Run the whole main.py.
   2. Compare results of different strategies for dealing with unclassified contigs:
      * In DNABERT-S they are simply discarded.
      * Assign unclassified contigs to nearest species centroid
      * Other assignment rules?
   3. Examine predictions and their embeddings across models. Good clusters should have a high inter-similarity and low intra-similarity. We want a metric that can be given to each contig, so we can make a histogram of the values and compare across models. Some suggestions:
      *  For each contig, calculate similarity with all species centroids to get matrix of (N, num_centroids). Treat these similarities as a probability distribution by applying softmax to each contig, then calculate entropy or get the argmax of these probabilities. Rationale: Good embeddings should have low entropy, or should have high argmax. 
      *  These metrics may not be so good, and we want to know if you have some ideas.  
   4. Plot histogram of similarities used for calculating the threshold for each model. 
   5. Make an outer loop going through the threshold percentile values and: 
      * See whether results are robust to changes in the percentile.
      * Are more species/contigs classified?
   6. Use other distance functions instead of np.dot on l2-normalised embeddings.   
   




##### Where do you need help from Veronika?

  * Point 2, 3, 4, 5, 6 above
  * We are very open to hear if you have some interresting experiments/vizualisations. 
  * Note that in our literature review, we found 17 DNA foundation models, and we got 6 of them to work at the moment. We know that we will not get all models up and running. 



