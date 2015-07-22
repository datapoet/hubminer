/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package learning.supervised.methods.knn;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import learning.supervised.interfaces.AutomaticKFinderInterface;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This class implements the Naive Hubness-Bayesian k-nearest neighbor algorithm
 * that was proposed in the following paper: A Probabilistic Approach to
 * Nearest-Neighbor Classification: Naive Hubness Bayesian kNN by Nenad Tomasev,
 * Milos Radovanovic, Dunja Mladenic and Mirjana Ivanovic, presented at the
 * Conference on Information and Knowledge Management (CIKM) in 2011 in Glasgow.
 * It is a Naive Bayesian interpretation of the k-nearest neighbor rule. Each
 * neighbor occurrence is interpreted as an event, as a new defining feature for
 * the query instance. Unlike in standard Naive Bayes, though - some of these
 * 'features', or rather feature values never occur on the training data and are
 * observed on the test data. This happens for points that are orphans in the
 * kNN graph on the training data and points like this arise frequently in
 * high-dimensional data due to the hubness phenomenon. Even for non-orphan
 * anti-hub points, deriving proper probability conditionals is difficult. On
 * the other hand, most neighbor occurrences in high-dimensional data are hub
 * occurrences and for those points it is possible to derive good
 * class-conditional occurrence probabilities and occurrence-conditional class
 * affiliation probabilities as well. In any case, it is possible to apply the
 * modified Naive Bayes rule for classification based on the kNN set. This
 * approach has later been shown to be quite promising in class-imbalanced
 * high-dimensional data. It was also a basis for development of ANHBNN that was
 * later presented at ECML/PKDD 2013 in Prague.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NHBNN extends Classifier implements AutomaticKFinderInterface,
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {

    private static final long serialVersionUID = 1L;
    // The default anti-hub threshold.
    private int thetaValue = 2;
    // Neighborhood size.
    private int k = 5;
    // The object that holds the kNN information.
    private NeighborSetFinder nsf = null;
    // The training dataset.
    private DataSet trainingData = null;
    // Number of classes in the data.
    private int numClasses = 0;
    // Class-conditional neighbor occurrence frequencies.
    private float[][] classDataKNeighborRelation = null;
    // Prior class distribution.
    private float[] classPriors = null;
    // The smoothing factor.
    private float laplaceEstimator = 0.05f;
    // Neighbor occurrence frequencies on the training data.
    private int[] neighbOccFreqs = null;
    // Several arrays for different types of local and global anti-hub vote
    // estimates, or in this case - conditional probability estimates and
    // approximations.
    private float[][][] localHClassDistribution = null;
    private float[][] classToClassPriors = null;
    // Upper triangular distance matrix.
    private float[][] distMat = null;
    // Variable holding the current estimate type.
    private int localEstimateMethod = GLOBAL;
    // The parameter that governs how much emphasis is put on the actual
    // occurrence counts for anti-hub estimates.
    private float alphaParam = 0.4f;
    // Estimation type constants.
    public static final int GLOBAL = 0;
    public static final int LOCALH = 1;
    private static final int K_LOCAL_APPROXIMATION = 20;
    private boolean noRecalc = false;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("thetaValue", "Anti-hub cut-off point for treating"
                + "anti-hubs as a special case.");
        paramMap.put("localEstimateMethod", "Anti-hub handling strategy.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("Conference on Information and Knowledge "
                + "Management");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.MILOS_RADOVANOVIC);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.addAuthor(Author.MIRJANA_IVANOVIC);
        pub.setTitle("A Probabilistic Approach to Nearest-Neighbor "
                + "Classification: Naive Hubness Bayesian kNN");
        pub.setYear(2011);
        pub.setStartPage(183);
        pub.setEndPage(195);
        pub.setPublisher(Publisher.SPRINGER);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "NHBNN";
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DataSet dset = trainingData;
        float currMaxAcc = -1f;
        int currMaxK = 0;
        int currMaxTheta = 0;
        float maxAlphaParam = 0.4f;
        int bestApproximationMethod = LOCALH;
        NHBNN classifier;
        ArrayList<DataInstance> data = dset.data;
        Random randa = new Random();
        ArrayList[] dataFolds = null;
        ArrayList currentTraining;
        ArrayList[] foldIndexes = null;
        ArrayList<Integer> currentIndexes;
        ArrayList<Integer> currentTest;
        // Generate folds for training and test.
        int folds = 2;
        float choice;
        boolean noEmptyFolds = false;
        while (!noEmptyFolds) {
            dataFolds = new ArrayList[folds];
            foldIndexes = new ArrayList[folds];
            for (int j = 0; j < folds; j++) {
                dataFolds[j] = new ArrayList(2000);
                foldIndexes[j] = new ArrayList<>(2000);
            }
            for (int j = 0; j < data.size(); j++) {
                choice = randa.nextFloat();
                if (choice < 0.15) {
                    dataFolds[1].add(data.get(j));
                    foldIndexes[1].add(j);
                } else {
                    dataFolds[0].add(data.get(j));
                    foldIndexes[0].add(j);
                }
            }
            // Check to see if some have remained empty, though it is highly
            // unlikely, since only 2 folds are used in this implementation.
            noEmptyFolds = true;
            for (int j = 0; j < folds; j++) {
                if (dataFolds[j].isEmpty()) {
                    noEmptyFolds = false;
                    break;
                }
            }
        }
        // Generate training and test datasets from the random folds.
        currentTest = foldIndexes[1];
        currentTraining = new ArrayList();
        currentIndexes = new ArrayList();
        currentTraining.addAll(dataFolds[0]);
        currentIndexes.addAll(foldIndexes[0]);
        classifier = (NHBNN) (copyConfiguration());
        classifier.setDataIndexes(currentIndexes, dset);
        ClassificationEstimator currEstimator;
        DataSet dsetTrain = dset.cloneDefinition();
        dsetTrain.data = currentTraining;
        NeighborSetFinder nsfAux;
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getDataIndexes(currentIndexes, dset);
        for (int i = 0; i < dsetTrain.size(); i++) {
            dsetTrain.data.set(i, dset.data.get(indexPermutation.get(i)));
        }
        // Prepare the sub-training kNN sets.
        if (distMat == null) {
            // Calculate the distance matrix from scratch if not already
            // available.
            nsfAux = new NeighborSetFinder(dsetTrain, getCombinedMetric());
            nsfAux.calculateDistances();
        } else {
            // Generate the fold distance matrix from the already provided
            // distance matrix.
            float[][] foldDistMatrix = new float[dsetTrain.size()][];
            int lowerIndex, upperIndex;
            for (int index1 = 0; index1 < foldDistMatrix.length; index1++) {
                foldDistMatrix[index1] =
                        new float[foldDistMatrix.length - index1 - 1];
                for (int index2 = index1 + 1; index2
                        < foldDistMatrix.length; index2++) {
                    lowerIndex = Math.min(indexPermutation.get(index1),
                            indexPermutation.get(index2));
                    upperIndex = Math.max(indexPermutation.get(index1),
                            indexPermutation.get(index2));
                    foldDistMatrix[index1][index2 - index1 - 1] =
                            distMat[lowerIndex][upperIndex - lowerIndex - 1];
                }
            }
            nsfAux = new NeighborSetFinder(dsetTrain, foldDistMatrix,
                    getCombinedMetric());
        }
        nsfAux.calculateNeighborSets(kMax);
        NeighborSetFinder nsfTIteration;
        // Iterate over different possible k-values.
        for (int kCurr = 1; kCurr <= kMax; kCurr++) {
            nsfTIteration = nsfAux.getSubNSF(kCurr);
            classifier.nsf = nsfTIteration;
            classifier.k = kCurr;
            classifier.train();
            // Iterate over different internal parameter values.
            for (thetaValue = 0; thetaValue < 10; thetaValue++) {
                for (alphaParam = 0; alphaParam <= 0.81; alphaParam += 0.2) {
                    // Attempt global anti-hub conditional probability
                    // approximation.
                    classifier.localEstimateMethod = GLOBAL;
                    classifier.thetaValue = thetaValue;
                    classifier.alphaParam = alphaParam;
                    currEstimator = classifier.test(currentTest, dset,
                            numClasses);
                    if (currEstimator.getAccuracy() > currMaxAcc) {
                        // If current best, save the parameter values.
                        currMaxAcc = currEstimator.getAccuracy();
                        currMaxK = kCurr;
                        currMaxTheta = thetaValue;
                        maxAlphaParam = alphaParam;
                        bestApproximationMethod = GLOBAL;
                    }
                    // Attempt local anti-hub conditional probability
                    // approximation.
                    classifier.localEstimateMethod = LOCALH;
                    classifier.thetaValue = thetaValue;
                    classifier.alphaParam = alphaParam;
                    currEstimator = classifier.test(currentTest, dset,
                            numClasses);
                    if (currEstimator.getAccuracy() > currMaxAcc) {
                        // If current best, save the parameter values.
                        currMaxAcc = currEstimator.getAccuracy();
                        currMaxK = kCurr;
                        currMaxTheta = thetaValue;
                        maxAlphaParam = alphaParam;
                        bestApproximationMethod = LOCALH;
                    }
                }
            }
        }
        // Set the best parameter values.
        thetaValue = currMaxTheta;
        k = currMaxK;
        alphaParam = maxAlphaParam;
        localEstimateMethod = bestApproximationMethod;
    }

    /**
     * The default constructor.
     */
    public NHBNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public NHBNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NHBNN(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public NHBNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     */
    public NHBNN(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NHBNN(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public NHBNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public NHBNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public NHBNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
            CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        this.nsf = nsf;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public NHBNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
    }

    /**
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimator = laplaceEstimator;
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        numClasses = trainingData.countCategories();
    }

    /**
     * @param trainingData DataSet object to train the model on.
     */
    public void setTrainingSet(DataSet trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @param numClasses Integer that is the number of classes in the data.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * @return Integer that is the number of classes in the data.
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * @return Integer that is the neighborhood size used in calculations.
     */
    public int getK() {
        return k;
    }

    /**
     * @param k Integer that is the neighborhood size used in calculations.
     */
    public void setK(int k) {
        this.k = k;
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    /**
     * Calculate the neighbor sets, if not already calculated.
     *
     * @throws Exception
     */
    public void calculateNeighborSets() throws Exception {
        if (distMat == null) {
            nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
            nsf.calculateDistances();
        } else {
            nsf = new NeighborSetFinder(trainingData, distMat,
                    getCombinedMetric());
        }
        nsf.calculateNeighborSets(k);
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        NHBNN classifierCopy = new NHBNN(k, laplaceEstimator,
                getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.localEstimateMethod = localEstimateMethod;
        classifierCopy.thetaValue = thetaValue;
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // If an invalid k-value is provided, calculate the appropriate
            // neighborhood size automatically from the default lower-end range.
            findK(1, 20);
        }
        if (nsf == null) {
            calculateNeighborSets();
        }
        // Get the class counts and the class priors.
        int[] classCounts = trainingData.getClassFrequencies();
        classPriors = trainingData.getClassPriors();
        // Get the total neighbor occurrence frequencies.
        neighbOccFreqs = nsf.getNeighborFrequencies();
        // Initialize the arrays for anti-hub approximations.
        localHClassDistribution =
                new float[trainingData.size()][numClasses][numClasses];
        // First fetch the kNN sets on the trainingdata.
        int[][] kneighbors = nsf.getKNeighbors();
        // Calculate the class-conditional occurrences.
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        float[][] distMatrix = nsf.getDistances();
        classToClassPriors = new float[numClasses][numClasses];
        for (int i = 0; i < trainingData.size(); i++) {
            int currClass = trainingData.data.get(i).getCategory();
            classDataKNeighborRelation[currClass][i]++;
            for (int kIndex = 0; kIndex < k; kIndex++) {
                classDataKNeighborRelation[currClass][kneighbors[i][kIndex]]++;
                classToClassPriors[trainingData.data.get(
                        kneighbors[i][kIndex]).getCategory()][currClass]++;
            }
            if (neighbOccFreqs[i] <= thetaValue) {
                // Anti-hub approximation is necessary.
                float[] localClassCounts = new float[numClasses];
                if (k >= K_LOCAL_APPROXIMATION) {
                    // If the neighborhood size is large enough for local
                    // approximation.
                    for (int kAppIndex = 0; kAppIndex < K_LOCAL_APPROXIMATION;
                            kAppIndex++) {
                        int currLClass = trainingData.data.get(kneighbors[i][
                                kAppIndex]).getCategory();
                        localClassCounts[currLClass]++;
                        localHClassDistribution[i][currLClass][currLClass]++;
                        for (int nIndex = 0; nIndex < k; nIndex++) {
                            localHClassDistribution[i][currLClass][
                                    trainingData.data.get(kneighbors[
                                    kneighbors[i][kAppIndex]][nIndex]).
                                    getCategory()]++;
                            // The first is the query class, the second the
                            // neighbor class.
                        }
                    }
                    // Normalize and smooth the approximation.
                    for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                        for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                            localHClassDistribution[i][cFirst][cSecond] +=
                                    (1f / numClasses);
                            localHClassDistribution[i][cFirst][cSecond] /=
                                    (localClassCounts[cFirst]
                                    * localClassCounts[cSecond]
                                    + (numClasses * (1f / numClasses)));
                        }
                    }

                } else {
                    // If the current neighborhood size is too small, calculate
                    // the remaining neighbor points around the query.
                    int[] localNeighbors = new int[K_LOCAL_APPROXIMATION];
                    float[] kDistances = new float[K_LOCAL_APPROXIMATION];
                    // Re-use what we already know.
                    for (int kIndex = 0; kIndex < k; kIndex++) {
                        localNeighbors[kIndex] = kneighbors[i][kIndex];
                        kDistances[kIndex] = nsf.getKDistances()[i][kIndex];
                    }
                    int kcurrLen = k;
                    int l;
                    float currDist;
                    boolean insertable;
                    for (int j = 0; j < trainingData.size(); j++) {
                        if (j != i) {
                            currDist = getDistanceForElements(distMatrix, i, j);
                            if (kcurrLen == K_LOCAL_APPROXIMATION) {
                                if (currDist < kDistances[kcurrLen - 1]) {
                                    // Search to see where to insert if it is
                                    // not already in present in the knn set.
                                    insertable = true;
                                    for (int index = 0; index < kcurrLen;
                                            index++) {
                                        if (j == localNeighbors[index]) {
                                            insertable = false;
                                            break;
                                        }
                                    }
                                    if (insertable) {
                                        l = kcurrLen - 1;
                                        while ((l >= 1) && currDist
                                                < kDistances[l - 1]) {
                                            kDistances[l] = kDistances[l - 1];
                                            localNeighbors[l] =
                                                    localNeighbors[l - 1];
                                            l--;
                                        }
                                        kDistances[l] = currDist;
                                        localNeighbors[l] = j;
                                    }
                                }
                            } else {
                                if (currDist < kDistances[kcurrLen - 1]) {
                                    // Search to see where to insert if it is
                                    // not already in present in the knn set.
                                    insertable = true;
                                    for (int index = 0; index < kcurrLen;
                                            index++) {
                                        if (j == localNeighbors[index]) {
                                            insertable = false;
                                            break;
                                        }
                                    }
                                    if (insertable) {
                                        l = kcurrLen - 1;
                                        kDistances[kcurrLen] =
                                                kDistances[kcurrLen - 1];
                                        localNeighbors[kcurrLen] =
                                                localNeighbors[kcurrLen - 1];
                                        while ((l >= 1) && currDist
                                                < kDistances[l - 1]) {
                                            kDistances[l] = kDistances[l - 1];
                                            localNeighbors[l] =
                                                    localNeighbors[l - 1];
                                            l--;
                                        }
                                        kDistances[l] = currDist;
                                        localNeighbors[l] = j;
                                        kcurrLen++;
                                    }
                                } else {
                                    kDistances[kcurrLen] = currDist;
                                    localNeighbors[kcurrLen] = j;
                                    kcurrLen++;
                                }
                            }
                        }
                    }
                    for (int kAppIndex = 0; kAppIndex < K_LOCAL_APPROXIMATION;
                            kAppIndex++) {
                        int currLClass = trainingData.data.get(
                                localNeighbors[kAppIndex]).getCategory();
                        localClassCounts[currLClass]++;
                        localHClassDistribution[i][currLClass][currLClass]++;
                        for (int nIndex = 0; nIndex < k; nIndex++) {
                            localHClassDistribution[i][currLClass][
                                    trainingData.data.get(kneighbors[
                                    localNeighbors[kAppIndex]][nIndex]).
                                    getCategory()]++;
                            // The first is the query class, the second the
                            // neighbor class.
                        }
                    }
                    // Normalize and smooth the approximation.
                    for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                        for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                            localHClassDistribution[i][cFirst][cSecond] +=
                                    (1f / (2 * numClasses));
                            localHClassDistribution[i][cFirst][cSecond] /=
                                    (localClassCounts[cFirst]
                                    * (K_LOCAL_APPROXIMATION + 1) + (numClasses
                                    * (1f / (2 * numClasses))));
                        }
                    }
                }
            }
        }
        // Normalize and smooth the class-conditional neighbor occurrence counts
        // into conditional probability estimates.
        float laplaceTotal = trainingData.size() * (1f / (2 * numClasses));
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < trainingData.size(); j++) {
                classDataKNeighborRelation[i][j] += (1f / (2 * numClasses));
                classDataKNeighborRelation[i][j] /= (k * (float) classCounts[i]
                        + laplaceTotal);
            }
        }
        // Normalize class-to-class priors.
        laplaceEstimator = 0.00001f;
        laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                classToClassPriors[i][j] += laplaceEstimator;
                classToClassPriors[i][j] /= ((k + 1) * (float) classCounts[j]
                        * (float) classCounts[i] + laplaceTotal);
            }
        }
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(reducer.getPrototypeIndexes(),
                reducer.getOriginalDataSet());
        int kReducer = reducer.getNeighborhoodSize();
        int[] protoNeighbOccFreqs = reducer.getPrototypeHubness();
        // Prototypes are the training set.
        // Get the class counts and the class priors.
        classPriors = trainingData.getClassPriors();
        localHClassDistribution =
                new float[trainingData.size()][numClasses][numClasses];
        // Get the unbiased prototype occurrence frequencies from the reducer.
        neighbOccFreqs = new int[protoNeighbOccFreqs.length];
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            neighbOccFreqs[i] = protoNeighbOccFreqs[indexPermutation.get(i)];
        }
        // Get the kNN sets with prototypes as neighbors from the reducer. We
        // must take care of the index permutation along the way.
        int[][] kneighbors = reducer.getProtoNeighborSets();
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        // The following call obtains the normalized distribution.
        float[][] classDataKNeighborRelationTemp = reducer.
                getClassDataNeighborRelationforBayesian(numClasses,
                laplaceEstimator);
        // Permute the values.
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            for (int i = 0; i < neighbOccFreqs.length; i++) {
                classDataKNeighborRelation[cIndex][i] =
                        classDataKNeighborRelationTemp[cIndex][
                        indexPermutation.get(i)];
            }
        }
        classToClassPriors = reducer.calculateClassToClassPriorsBayesian();
        int actualIndex;
        for (int i = 0; i < trainingData.size(); i++) {
            actualIndex = indexPermutation.get(i);
            if (neighbOccFreqs[i] <= thetaValue) {
                // The anti-hub approximation case.
                float[] localClassCounts = new float[numClasses];
                for (int kIndex = 0; kIndex < kReducer; kIndex++) {
                    int currLClass = reducer.getPrototypeLabel(
                            kneighbors[actualIndex][kIndex]);
                    localClassCounts[currLClass]++;
                    localHClassDistribution[i][currLClass][currLClass]++;
                    for (int nIndex = 0; nIndex < kReducer; nIndex++) {
                        localHClassDistribution[i][currLClass][reducer.
                                getPrototypeLabel(kneighbors[kneighbors[
                                actualIndex][kIndex]][nIndex])]++;
                        // The first is the query class, the second the
                        // neighbor class.
                    }
                }
                for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                    for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                        localHClassDistribution[i][cFirst][cSecond] +=
                                (1f / numClasses);
                        localHClassDistribution[i][cFirst][cSecond] /=
                                (localClassCounts[cFirst] * localClassCounts[
                                cSecond] + (numClasses * (1f / numClasses)));
                    }
                }
            }
        }
    }

    /**
     * Gets the distance between the two instances from the upper triangular
     * distance matrix.
     *
     * @param distMatrix float[][] that is the upper triangular distance matrix.
     * @param i Integer that is the index of the first instance.
     * @param j Integer that is the index of the second instance.
     * @return Float value that is the distance between the two instances.
     */
    private static float getDistanceForElements(float[][] distMatrix, int i,
            int j) {
        if (j > i) {
            return distMatrix[i][j - i - 1];
        } else if (i > j) {
            return distMatrix[j][i - j - 1];
        } else {
            return 0;
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
        float maxProb = 0;
        int maxClass = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClass = cIndex;
            }
        }
        return maxClass;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        // Calculate the kNN set.
        float[] kDistances = new float[k];
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDistance = cmet.dist(trainingData.data.get(i), instance);
            if (currDistance < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > currDistance) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = currDistance;
                kNeighbors[index] = i;
            }
        }
        double[] classProbEstimates = new double[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            classProbEstimates[cIndex] = classPriors[cIndex];
        }
        double maxProb = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighbOccFreqs[kNeighbors[kIndex]] > thetaValue) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] *= classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]];
                    if (classProbEstimates[cIndex] > maxProb) {
                        maxProb = classProbEstimates[cIndex];
                    }
                }

            } else {
                float occTotal = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    occTotal += classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]];
                }
                float globalEstimateDenominator = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    globalEstimateDenominator += classToClassPriors[
                            trainingData.data.get(kNeighbors[kIndex]).
                            getCategory()][cIndex];
                }
                float localEstimateDenominator = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    localEstimateDenominator += localHClassDistribution[
                            kNeighbors[kIndex]][cIndex][trainingData.data.get(
                            kNeighbors[kIndex]).getCategory()];
                }
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    float gFact = (alphaParam * classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]]) / occTotal
                            + ((1 - alphaParam) * classToClassPriors[
                            trainingData.data.get(kNeighbors[kIndex]).
                            getCategory()][cIndex])
                            / globalEstimateDenominator;
                    float lhFact = (alphaParam * classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]]) / occTotal
                            + ((1 - alphaParam) * localHClassDistribution[
                            kNeighbors[kIndex]][cIndex][trainingData.data.get(
                            kNeighbors[kIndex]).getCategory()])
                            / localEstimateDenominator;
                    // Some correction is used to avoid floating-point
                    // issues in case the number of neighbors is too large.
                    switch (localEstimateMethod) {
                        case GLOBAL:
                            classProbEstimates[cIndex] *= 10 * gFact;
                            break;
                        case LOCALH:
                            classProbEstimates[cIndex] *= 10 * lhFact;
                        default:
                            classProbEstimates[cIndex] *= 10 * lhFact;
                    }
                    if (classProbEstimates[cIndex] > maxProb) {
                        maxProb = classProbEstimates[cIndex];
                    }
                }
            }
            if (maxProb > 0) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] /= maxProb;
                }
            }
        }
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] = classPriors[cIndex];
            }
        }
        float[] floatProbEstimates = new float[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            floatProbEstimates[cIndex] = (float) classProbEstimates[cIndex];
        }
        return floatProbEstimates;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Calculate the kNN set.
        float[] kDistances = new float[k];
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDistance = distToTraining[i];
            if (currDistance < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > currDistance) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = currDistance;
                kNeighbors[index] = i;
            }
        }
        double[] classProbEstimates = new double[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            classProbEstimates[cIndex] = classPriors[cIndex];
        }
        double maxProb = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighbOccFreqs[kNeighbors[kIndex]] > thetaValue) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] *= classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]];
                    if (classProbEstimates[cIndex] > maxProb) {
                        maxProb = classProbEstimates[cIndex];
                    }
                }
            } else {
                float occTotal = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    occTotal += classDataKNeighborRelation[cIndex][kNeighbors[
                            kIndex]];
                }
                float globalEstimateDenominator = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    globalEstimateDenominator += classToClassPriors[
                            trainingData.data.get(kNeighbors[kIndex]).
                            getCategory()][cIndex];
                }
                float localEstimateDenominator = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    localEstimateDenominator += localHClassDistribution[
                            kNeighbors[kIndex]][cIndex][trainingData.data.get(
                            kNeighbors[kIndex]).getCategory()];
                }
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    float gFact = (alphaParam * classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]]) / occTotal
                            + ((1 - alphaParam) * classToClassPriors[
                            trainingData.data.get(kNeighbors[kIndex]).
                            getCategory()][cIndex]) / globalEstimateDenominator;
                    float lhFact = (alphaParam * classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]]) / occTotal
                            + ((1 - alphaParam) * localHClassDistribution[
                            kNeighbors[kIndex]][cIndex][trainingData.data.get(
                            kNeighbors[kIndex]).getCategory()])
                            / localEstimateDenominator;
                    switch (localEstimateMethod) {
                        // Some correction is used to avoid floating-point
                        // issues in case the number of neighbors is too large.
                        case GLOBAL:
                            classProbEstimates[cIndex] *= 10 * gFact;
                            break;
                        case LOCALH:
                            classProbEstimates[cIndex] *= 10 * lhFact;
                        default:
                            classProbEstimates[cIndex] *= 10 * lhFact;
                    }
                    if (classProbEstimates[cIndex] > maxProb) {
                        maxProb = classProbEstimates[cIndex];
                    }
                }
            }
            if (maxProb > 0) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] /= maxProb;
                }
            }
        }
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] = classPriors[cIndex];
            }
        }
        float[] floatProbEstimates = new float[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            floatProbEstimates[cIndex] = (float) classProbEstimates[cIndex];
        }
        return floatProbEstimates;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining,
            int[] trNeighbors) throws Exception {
        float[] classProbs = classifyProbabilistically(instance, distToTraining,
                trNeighbors);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining, int[] trNeighbors) throws Exception {
        double[] classProbEstimates = new double[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            classProbEstimates[cIndex] = classPriors[cIndex];
        }
        double maxProb = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighbOccFreqs[trNeighbors[kIndex]] > thetaValue) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] *= classDataKNeighborRelation[
                            cIndex][trNeighbors[kIndex]];
                    if (classProbEstimates[cIndex] > maxProb) {
                        maxProb = classProbEstimates[cIndex];
                    }
                }
            } else {
                float occTotal = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    occTotal += classDataKNeighborRelation[cIndex][
                            trNeighbors[kIndex]];
                }
                float globalEstimateDenominator = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    globalEstimateDenominator += classToClassPriors[
                            trainingData.data.get(trNeighbors[kIndex]).
                            getCategory()][cIndex];
                }
                float localEstimateDenominator = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    localEstimateDenominator += localHClassDistribution[
                            trNeighbors[kIndex]][cIndex][trainingData.data.get(
                            trNeighbors[kIndex]).getCategory()];
                }
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    float gFact = (alphaParam * classDataKNeighborRelation[
                            cIndex][trNeighbors[kIndex]]) / occTotal
                            + ((1 - alphaParam) * classToClassPriors[
                            trainingData.data.get(trNeighbors[kIndex]).
                            getCategory()][cIndex]) / globalEstimateDenominator;
                    float lhFact = (alphaParam * classDataKNeighborRelation[
                            cIndex][trNeighbors[kIndex]]) / occTotal
                            + ((1 - alphaParam) * localHClassDistribution[
                            trNeighbors[kIndex]][cIndex][trainingData.data.get(
                            trNeighbors[kIndex]).getCategory()])
                            / localEstimateDenominator;
                    switch (localEstimateMethod) {
                        // Some correction is used to avoid floating-point
                        // issues in case the number of neighbors is too large.
                        case GLOBAL:
                            classProbEstimates[cIndex] *= 10 * gFact;
                            break;
                        case LOCALH:
                            classProbEstimates[cIndex] *= 10 * lhFact;
                        default:
                            classProbEstimates[cIndex] *= 10 * lhFact;

                    }
                    if (classProbEstimates[cIndex] > maxProb) {
                        maxProb = classProbEstimates[cIndex];
                    }
                }
            }
            if (maxProb > 0) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] /= maxProb;
                }
            }
        }
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] = classPriors[cIndex];
            }
        }
        float[] floatProbEstimates = new float[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            floatProbEstimates[cIndex] = (float) classProbEstimates[cIndex];
        }
        return floatProbEstimates;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
