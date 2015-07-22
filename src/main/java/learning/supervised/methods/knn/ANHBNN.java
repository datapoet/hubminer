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
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.awt.Point;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;
import util.BasicMathUtil;

/**
 * This class implements the Augmented Naive Hubness-Bayesian k-Nearest Neighbor
 * method that was proposed in the following paper: Hub Co-occurrence Modeling
 * for Robust High-dimensional kNN Classification, by Nenad Tomasev and Dunja
 * Mladenic, which was presented at ECML/PKDD 2013 in Prague. The algorithm is
 * an extension of NHBNN, the Naive Bayesian re-interpretation of the k-nearest
 * neighbor rule. It incorporates the Hidden Naive Bayes model for kNN
 * classification by modeling the class-conditional neighbor co-occurrence
 * probabilities.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ANHBNN extends Classifier implements DistMatrixUserInterface,
        NSFUserInterface, DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    // The anti-hub cut-off parameter.
    private int thetaValue = 0;
    // The current neighborhood size.
    private int k = 5;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;
    // Training dataset to infer the model from.
    private DataSet trainingData = null;
    // The number of classes in the data.
    private int numClasses = 0;
    private ArrayList<Point> coOccurringPairs;
    // One map for each class, that maps a long value obtained from two index
    // values to a value that is the current co-occurrence count.
    private HashMap<Long, Integer>[] coDependencyMaps;
    // Mutual information between neighbor pairs.
    private HashMap<Long, Double> mutualInformationMap;
    // Class-conditional neighbor occurrence counts.
    private float[][] classDataKNeighborRelation = null;
    // Class priors.
    private float[] classPriors = null;
    // Class frequencies.
    private float[] classFreqs = null;
    // Float value that is the Laplace estimator for probability distribution
    // smoothing. There are two values, a smaller and a larger one, for
    // smoothing different types of distributions.
    private float laplaceEstimatorSmall = 0.000000000001f;
    private float laplaceEstimatorBig = 0.1f;
    // Neighbor occurrence frequencies.
    private int[] neighbOccFreqs = null;
    // Non-homogeneity of reverse neighbor sets.
    private float[] rnnImpurity = null;
    private double[][] classConditionalSelfInformation = null;
    private float[][] classToClassPriors = null;
    private float[][][] classCoOccurrencesInNeighborhoodsOfClasses;
    private float[][] distMat = null;
    private boolean noRecalc = false;
    private int dataSize;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("thetaValue", "Anti-hub cut-off point for treating"
                + "anti-hubs as a special case.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("European Conference on Machine Learning");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.setTitle("Hub Co-occurrence Modeling for Robust High-dimensional "
                + "kNN Classification");
        pub.setYear(2013);
        pub.setStartPage(643);
        pub.setEndPage(659);
        pub.setPublisher(Publisher.SPRINGER);
        return pub;
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
    public String getName() {
        return "ANHBNN";
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    /**
     * The default constructor.
     */
    public ANHBNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public ANHBNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public ANHBNN(int k, CombinedMetric cmet) {
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
    public ANHBNN(int k, CombinedMetric cmet, int numClasses) {
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
    public ANHBNN(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimatorSmall = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public ANHBNN(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimatorSmall = laplaceEstimator;
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
    public ANHBNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimatorSmall = laplaceEstimator;
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
    public ANHBNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
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
    public ANHBNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
    public ANHBNN(Category[] categories, CombinedMetric cmet, int k) {
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
        this.laplaceEstimatorSmall = laplaceEstimator;
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
        ANHBNN classifierCopy = new ANHBNN(k, laplaceEstimatorSmall,
                getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.thetaValue = thetaValue;
        return classifierCopy;
    }

    /**
     * Calculates the mutual information between two neighbors according to
     * their occurrences and co-occurrences.
     *
     * @param min
     * @param max
     * @return
     */
    private double calculateMutualInformation(int lowerIndex, long upperIndex) {
        // Transform to the encoding used in the hash maps.
        long concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
        int size = trainingData.size();
        if (mutualInformationMap.containsKey(concat)) {
            // If it has already been queried and calculated before, just load
            // the existing result.
            return mutualInformationMap.get(concat);
        } else {
            // Calculate the mutual information from mutual and individual
            // occurrence counts.
            double bothOccurFactor = 0;
            double firstOccursFactor = 0;
            double secondOccursFactor = 0;
            double noneOccursFactor = 0;
            int cooccFreq;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (coDependencyMaps[cIndex].containsKey(concat)) {
                    cooccFreq = coDependencyMaps[cIndex].get(concat);
                } else {
                    cooccFreq = 0;
                }
                // The formulas are a bit complicated. For more detail, look up
                // the original paper, as it is freely available online.
                bothOccurFactor += ((double) (cooccFreq) / (double) size)
                        * BasicMathUtil.log2(((double) (cooccFreq
                        + laplaceEstimatorSmall) / (double) classFreqs[cIndex]
                        + laplaceEstimatorSmall) / (((classDataKNeighborRelation[
                        cIndex][lowerIndex] + laplaceEstimatorSmall)
                        / ((double) classFreqs[cIndex] + laplaceEstimatorSmall))
                        * ((classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall))));
                firstOccursFactor += ((double) (classDataKNeighborRelation[
                        cIndex][lowerIndex] - cooccFreq) / (double) size)
                        * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[cIndex][lowerIndex]
                        - cooccFreq + laplaceEstimatorSmall) /
                        ((double) classFreqs[cIndex] + laplaceEstimatorSmall))
                        / (((classDataKNeighborRelation[cIndex][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)) * (1
                        - ((classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)))));
                secondOccursFactor += ((double) (classDataKNeighborRelation[
                        cIndex][(int) upperIndex] - cooccFreq) / (double) size)
                        * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[cIndex][(int) upperIndex]
                        - cooccFreq + laplaceEstimatorSmall) /
                        ((double) classFreqs[cIndex] + laplaceEstimatorSmall)) /
                        (((classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)) * (1 -
                        ((classDataKNeighborRelation[cIndex][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)))));
                noneOccursFactor += ((double) (classFreqs[cIndex]
                        - classDataKNeighborRelation[cIndex][lowerIndex]
                        - classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + cooccFreq) / (double) size) * BasicMathUtil.log2(
                        ((double) (classFreqs[cIndex]
                        - classDataKNeighborRelation[cIndex][lowerIndex]
                        - classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + cooccFreq + laplaceEstimatorSmall)
                        / ((double) classFreqs[cIndex] + laplaceEstimatorSmall))
                        / ((1 - ((classDataKNeighborRelation[cIndex][
                        (int) upperIndex] + laplaceEstimatorSmall)
                        / ((double) classFreqs[cIndex] +
                        laplaceEstimatorSmall)))
                        * (1 - ((classDataKNeighborRelation[cIndex][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)))));
            }
            double mutualInformation = bothOccurFactor + firstOccursFactor
                    + secondOccursFactor + noneOccursFactor;
            mutualInformationMap.put(concat, mutualInformation);
            return mutualInformation;
        }
    }

    /**
     * Calculates the entropies of the reverse neighbor sets.
     *
     * @param kneighbors int[][] of k-nearest neighbors for all training
     * instances.
     * @return float[] of reverse neighbor set entropies for all training
     * instances.
     */
    private float[] calculateReverseNeighborEntropies(int[][] kneighbors) {
        // Category frequencies in the reverse neighbor set of a particular
        // point.
        float[] categoryFrequencies = new float[numClasses];
        float[] reverseNeighborEntropies = new float[trainingData.size()];
        ArrayList<Integer>[] reverseNeighbors =
                new ArrayList[trainingData.size()];
        for (int i = 0; i < trainingData.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        for (int i = 0; i < trainingData.size(); i++) {
            for (int kIndex = 0; kIndex < k; kIndex++) {
                reverseNeighbors[kneighbors[i][kIndex]].add(i);
            }
        }
        float ratio;
        for (int i = 0; i < reverseNeighborEntropies.length; i++) {
            if (reverseNeighbors[i].size() <= 1) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    categoryFrequencies[cIndex] = 0;
                }
                reverseNeighborEntropies[i] = 0;
                continue;
            }
            for (int j = 0; j < reverseNeighbors[i].size(); j++) {
                int currClass = trainingData.getInstance(
                        reverseNeighbors[i].get(j)).getCategory();
                if (currClass >= 0) {
                    categoryFrequencies[currClass]++;
                }
            }
            // Calculate the entropy.
            for (int j = 0; j < categoryFrequencies.length; j++) {
                if (categoryFrequencies[j] > 0) {
                    ratio = categoryFrequencies[j]
                            / (float) reverseNeighbors[i].size();
                    reverseNeighborEntropies[i] -=
                            ratio * BasicMathUtil.log2(ratio);
                }
            }
            // Nullify the category frequencies array for next use.
            for (int j = 0; j < numClasses; j++) {
                categoryFrequencies[j] = 0;
            }
        }
        return reverseNeighborEntropies;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // If the neighborhood size was not specified, use the default
            // value. TODO: implement automatic parameter selection.
            k = 10;
        }
        if (nsf == null) {
            calculateNeighborSets();
        }
        dataSize = trainingData.size();
        // Calculate class priors.
        classPriors = trainingData.getClassPriors();
        classFreqs = trainingData.getClassFrequenciesAsFloatArray();
        classToClassPriors = new float[numClasses][numClasses];
        // Get the neighbor occurrence frequencies.
        neighbOccFreqs = nsf.getNeighborFrequencies();
        // Calculate the entropies of the reverse neighbor sets.
        nsf.calculateReverseNeighborEntropies(numClasses);
        rnnImpurity = nsf.getReverseNeighborEntropies();
        // The list of co-occurring pairs of points.
        coOccurringPairs = new ArrayList<>(dataSize);
        // The kNN sets.
        int[][] kneighbors = nsf.getKNeighbors();
        // The map for storing the co-occurrence counts. 
        coDependencyMaps = new HashMap[numClasses];
        mutualInformationMap = new HashMap<>(dataSize);
        classConditionalSelfInformation = new double[dataSize][numClasses];
        classCoOccurrencesInNeighborhoodsOfClasses =
                new float[numClasses][numClasses][numClasses];
        // Initialize the maps.
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            coDependencyMaps[cIndex] = new HashMap<>(dataSize);

        }
        long concat; // Used to encode neighbor pairs to a single hashable
        // value.
        int lowerIndex;
        long upperIndex;
        int queryClass;
        int currFreq;
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            // Get the class context of the query.
            queryClass = trainingData.getLabelOf(i);
            // Each point is considered as its own 0th nearest neighbor, which
            // avoids many zero-division issues.
            classDataKNeighborRelation[queryClass][i]++;
            classToClassPriors[queryClass][queryClass]++;
            for (int kIndFirst = 0; kIndFirst < k; kIndFirst++) {
                classDataKNeighborRelation[queryClass][kneighbors[i][
                        kIndFirst]]++;
                classToClassPriors[trainingData.getLabelOf(
                        kneighbors[i][kIndFirst])][queryClass]++;
                // Encode the pair. Here each neighbor co-occurs with the query
                // point, as the point is considered to be its own neighbor.
                lowerIndex = Math.min(kneighbors[i][kIndFirst], i);
                upperIndex = Math.max(kneighbors[i][kIndFirst], i);
                concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                // Insert or increment the co-occurrence count.
                if (!coDependencyMaps[queryClass].containsKey(concat)) {
                    coDependencyMaps[queryClass].put(concat, 1);
                    coOccurringPairs.add(new Point(lowerIndex,
                            (int) upperIndex));
                } else {
                    currFreq = coDependencyMaps[queryClass].get(concat);
                    coDependencyMaps[queryClass].remove(concat);
                    coDependencyMaps[queryClass].put(concat, currFreq + 1);
                }
                // Iterate through the remaining neighbors.
                for (int kIndSecond = kIndFirst + 1; kIndSecond < k;
                        kIndSecond++) {
                    classCoOccurrencesInNeighborhoodsOfClasses[queryClass][
                            trainingData.getLabelOf(kneighbors[i][kIndFirst])][
                            trainingData.getLabelOf(
                            kneighbors[i][kIndSecond])]++;
                    if (trainingData.getLabelOf(kneighbors[i][kIndFirst])
                            != trainingData.getLabelOf(
                            kneighbors[i][kIndSecond])) {
                        classCoOccurrencesInNeighborhoodsOfClasses[queryClass][
                                trainingData.getLabelOf(kneighbors[i][
                                kIndSecond])][trainingData.getLabelOf(
                                kneighbors[i][kIndFirst])]++;
                    }
                    // Encode the pair.
                    lowerIndex = Math.min(kneighbors[i][kIndFirst],
                            kneighbors[i][kIndSecond]);
                    upperIndex = Math.max(kneighbors[i][kIndFirst],
                            kneighbors[i][kIndSecond]);
                    concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                    // Insert or increment the co-occurrence count.
                    if (!coDependencyMaps[queryClass].containsKey(concat)) {
                        coDependencyMaps[queryClass].put(concat, 1);
                        coOccurringPairs.add(new Point(lowerIndex,
                                (int) upperIndex));
                    } else {
                        currFreq = coDependencyMaps[queryClass].get(concat);
                        coDependencyMaps[queryClass].remove(concat);
                        coDependencyMaps[queryClass].put(concat, currFreq + 1);
                    }
                }
            }
        }
        // Calculate the class-conditional self-information.
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            for (int c = 0; c < numClasses; c++) {
                if (classDataKNeighborRelation[c][i] > 0) {
                    classConditionalSelfInformation[i][c] =
                            BasicMathUtil.log2(classFreqs[c]
                            / classDataKNeighborRelation[c][i]);
                } else {
                    classConditionalSelfInformation[i][c] =
                            BasicMathUtil.log2(classFreqs[c]);
                }
            }
        }
        double bothOccurFactor;
        double firstOccursFactor;
        double secondOccursFactor;
        double noneOccursFactor;
        int cooccFreq;
        // Mutual information calculations.
        for (int i = 0; i < coOccurringPairs.size(); i++) {
            bothOccurFactor = 0;
            firstOccursFactor = 0;
            secondOccursFactor = 0;
            noneOccursFactor = 0;
            lowerIndex = (int) (coOccurringPairs.get(i).getX());
            upperIndex = (int) (coOccurringPairs.get(i).getY());
            // Encode the pair.
            concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
            for (int c = 0; c < numClasses; c++) {
                if (coDependencyMaps[c].containsKey(concat)) {
                    cooccFreq = coDependencyMaps[c].get(concat);
                } else {
                    cooccFreq = 0;
                }
                bothOccurFactor += ((double) (cooccFreq) / (double) dataSize)
                        * BasicMathUtil.log2(((double) (cooccFreq
                        + laplaceEstimatorSmall) / (double) classFreqs[c]
                        + laplaceEstimatorSmall)
                        / (((classDataKNeighborRelation[c][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall))
                        * ((classDataKNeighborRelation[c][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall))));
                firstOccursFactor += ((double) (
                        classDataKNeighborRelation[c][lowerIndex] - cooccFreq)
                        / (double) dataSize) * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[c][lowerIndex] - cooccFreq
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall))
                        / (((classDataKNeighborRelation[c][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)) * (1
                        - ((classDataKNeighborRelation[c][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)))));
                secondOccursFactor += ((double) (classDataKNeighborRelation[c][
                        (int) upperIndex] - cooccFreq) / (double) dataSize)
                        * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[c][(int) upperIndex]
                        - cooccFreq + laplaceEstimatorSmall) / (
                        (double) classFreqs[c] + laplaceEstimatorSmall)) /
                        (((classDataKNeighborRelation[c][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)) * (1 - ((
                        classDataKNeighborRelation[c][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)))));
                noneOccursFactor += ((double) (classFreqs[c]
                        - classDataKNeighborRelation[c][lowerIndex]
                        - classDataKNeighborRelation[c][(int) upperIndex]
                        + cooccFreq) / (double) dataSize) * BasicMathUtil.log2(
                        ((double) (classFreqs[c] - classDataKNeighborRelation[
                        c][lowerIndex] - classDataKNeighborRelation[c][
                        (int) upperIndex] + cooccFreq + laplaceEstimatorSmall)
                        / ((double) classFreqs[c] + laplaceEstimatorSmall))
                        / ((1 - ((classDataKNeighborRelation[c][
                        (int) upperIndex] + laplaceEstimatorSmall) /
                        ((double) classFreqs[c] + laplaceEstimatorSmall))) * (1
                        - ((classDataKNeighborRelation[c][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)))));
            }
            mutualInformationMap.put(concat, bothOccurFactor
                    + firstOccursFactor + secondOccursFactor +
                    noneOccursFactor);
        }
        // Normalize class-to-class priors.
        laplaceEstimatorSmall = 0.00001f;
        double laplaceTotal = numClasses * laplaceEstimatorSmall;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimatorSmall;
                classToClassPriors[cFirst][cSecond] /= ((k + 1)
                        * classFreqs[cSecond] * classFreqs[cFirst]
                        + laplaceTotal);
            }
        }
        // Now we calculate how often classes co-occur in neighborhoods of each
        // particular class.
        float occSum;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                occSum = 0;
                for (int cThird = 0; cThird < numClasses; cThird++) {
                    occSum += classCoOccurrencesInNeighborhoodsOfClasses[
                            cFirst][cSecond][cThird];
                }
                if (occSum > 0) {
                    for (int cThird = 0; cThird < numClasses; cThird++) {
                        classCoOccurrencesInNeighborhoodsOfClasses[
                                cFirst][cSecond][cThird] /= occSum;
                    }
                }
            }
        }
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(reducer.getPrototypeIndexes(),
                reducer.getOriginalDataSet());
        int kReducer = reducer.getNeighborhoodSize();
        if (kReducer <= 0) {
            kReducer = 10;
        }
        if (nsf == null) {
            calculateNeighborSets();
        }
        dataSize = reducer.getOriginalDataSize();
        // First find the class priors.
        classPriors = trainingData.getClassPriors();
        classFreqs = trainingData.getClassFrequenciesAsFloatArray();
        classToClassPriors = new float[numClasses][numClasses];
        // Get the neighbor occurrence frequencies.
        int[] protoOccFreqs = reducer.getPrototypeHubness();
        neighbOccFreqs = new int[protoOccFreqs.length];
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            neighbOccFreqs[i] = protoOccFreqs[indexPermutation.get(i)];
        }
        // Initialize class-to-class priors.
        classToClassPriors = new float[numClasses][numClasses];
        // Initialize the list of co-occuring neighbor pairs.
        coOccurringPairs = new ArrayList<>(trainingData.size());
        // Get the prototype kNN sets.
        int[][] kneighbors = reducer.getProtoNeighborSets();
        // Calculate the reverse neighbor entropies.
        rnnImpurity = this.calculateReverseNeighborEntropies(kneighbors);
        // Initialize the hash maps.
        coDependencyMaps = new HashMap[numClasses];
        mutualInformationMap = new HashMap<>(trainingData.size());
        classConditionalSelfInformation = new double[trainingData.size()][
                numClasses];
        classCoOccurrencesInNeighborhoodsOfClasses =
                new float[numClasses][numClasses][numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            coDependencyMaps[cIndex] = new HashMap<>(trainingData.size());
        }
        float[][] classDataKNeighborRelationTemp = reducer.
                getClassDataNeighborRelationNonNormalized(
                kReducer, numClasses, true);
        classDataKNeighborRelation = new float[numClasses][
                neighbOccFreqs.length];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            for (int i = 0; i < neighbOccFreqs.length; i++) {
                classDataKNeighborRelation[cIndex][i] =
                        classDataKNeighborRelationTemp[cIndex][
                        indexPermutation.get(i)];
            }
        }
        // Encoding for the neighbor pairs.
        long concat;
        int lowerIndex;
        long upperIndex;
        int queryClass;
        int currFreq;
        int actualIndex;
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            actualIndex = indexPermutation.get(i);
            queryClass = trainingData.getLabelOf(i);
            classToClassPriors[queryClass][queryClass]++;
            for (int kIndFirst = 0; kIndFirst < kReducer; kIndFirst++) {
                // Get the encoding for co-occurrences of the query point that
                // is considered to be its own zeroeth neighbor and the
                // prototype.
                lowerIndex = Math.min(kneighbors[actualIndex][kIndFirst], i);
                upperIndex = Math.max(kneighbors[actualIndex][kIndFirst], i);
                concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                if (!coDependencyMaps[queryClass].containsKey(concat)) {
                    coDependencyMaps[queryClass].put(concat, 1);
                    coOccurringPairs.add(new Point(lowerIndex,
                            (int) upperIndex));
                } else {
                    currFreq = coDependencyMaps[queryClass].get(concat);
                    coDependencyMaps[queryClass].remove(concat);
                    coDependencyMaps[queryClass].put(concat, currFreq + 1);
                }
                classToClassPriors[trainingData.getLabelOf(
                        kneighbors[actualIndex][kIndFirst])][queryClass]++;
                // Consider neighbor co-occurrences.
                for (int kIndSecond = kIndFirst + 1; kIndSecond < kReducer;
                        kIndSecond++) {
                    // Increment the class-to-class co-occurrence counts in
                    // neighborhoods of the current query class.
                    classCoOccurrencesInNeighborhoodsOfClasses[queryClass][
                            trainingData.getLabelOf(kneighbors[i][kIndFirst])][
                            trainingData.getLabelOf(
                            kneighbors[i][kIndSecond])]++;
                    if (trainingData.getLabelOf(kneighbors[i][kIndFirst])
                            != trainingData.getLabelOf(
                            kneighbors[i][kIndSecond])) {
                        classCoOccurrencesInNeighborhoodsOfClasses[queryClass][
                                trainingData.getLabelOf(kneighbors[i][
                                kIndSecond])][trainingData.getLabelOf(
                                kneighbors[i][kIndFirst])]++;
                    }
                    // Encode the pair from their indexes.
                    lowerIndex = Math.min(kneighbors[actualIndex][kIndFirst],
                            kneighbors[actualIndex][kIndSecond]);
                    upperIndex = Math.max(kneighbors[actualIndex][kIndFirst],
                            kneighbors[actualIndex][kIndSecond]);
                    concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                    // Insert or increment the co-occurrence count for the
                    // neighbor pair.
                    if (!coDependencyMaps[queryClass].containsKey(concat)) {
                        coDependencyMaps[queryClass].put(concat, 1);
                        coOccurringPairs.add(new Point(lowerIndex,
                                (int) upperIndex));
                    } else {
                        currFreq = coDependencyMaps[queryClass].get(concat);
                        coDependencyMaps[queryClass].remove(concat);
                        coDependencyMaps[queryClass].put(concat, currFreq + 1);
                    }
                }
            }
        }
        // Calculate the class-conditional occurrence self-information.
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (classDataKNeighborRelation[cIndex][i] > 0) {
                    classConditionalSelfInformation[i][cIndex] =
                            BasicMathUtil.log2(classFreqs[cIndex]
                            / classDataKNeighborRelation[cIndex][i]);
                } else {
                    classConditionalSelfInformation[i][cIndex] = 0;
                }
            }
        }
        double bothOccurFactor;
        double firstOccursFactor;
        double secondOccursFactor;
        double noneOccursFactor;
        int cooccFreq;
        // Mutual information calculations.
        for (int i = 0; i < coOccurringPairs.size(); i++) {
            bothOccurFactor = 0;
            firstOccursFactor = 0;
            secondOccursFactor = 0;
            noneOccursFactor = 0;
            // Encode the pair.
            lowerIndex = (int) (coOccurringPairs.get(i).getX());
            upperIndex = (int) (coOccurringPairs.get(i).getY());
            concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
            for (int c = 0; c < numClasses; c++) {
                if (coDependencyMaps[c].containsKey(concat)) {
                    cooccFreq = coDependencyMaps[c].get(concat);
                } else {
                    cooccFreq = 0;
                }
                bothOccurFactor += ((double) (cooccFreq) / (double) dataSize)
                        * BasicMathUtil.log2(((double) (cooccFreq
                        + laplaceEstimatorSmall) / (double) classFreqs[c]
                        + laplaceEstimatorSmall) /
                        (((classDataKNeighborRelation[c][lowerIndex] +
                        laplaceEstimatorSmall) / ((double) classFreqs[c] +
                        laplaceEstimatorSmall)) * ((classDataKNeighborRelation[
                        c][(int) upperIndex] + laplaceEstimatorSmall) /
                        ((double) classFreqs[c] + laplaceEstimatorSmall))));
                firstOccursFactor += ((double) (classDataKNeighborRelation[c][
                        lowerIndex] - cooccFreq) / (double) dataSize)
                        * BasicMathUtil.log2(((double)
                        (classDataKNeighborRelation[c][lowerIndex] - cooccFreq
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)) / (((classDataKNeighborRelation[
                        c][lowerIndex] + laplaceEstimatorSmall) /
                        ((double) classFreqs[c] + laplaceEstimatorSmall)) * (1
                        - ((classDataKNeighborRelation[c][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)))));
                secondOccursFactor += ((double) (classDataKNeighborRelation[c][
                        (int) upperIndex] - cooccFreq) / (double) dataSize)
                        * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[c][(int) upperIndex]
                        - cooccFreq + laplaceEstimatorSmall) / (
                        (double) classFreqs[c] + laplaceEstimatorSmall)) /
                        (((classDataKNeighborRelation[c][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)) * (1 - (
                        (classDataKNeighborRelation[c][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)))));
                noneOccursFactor += ((double) (classFreqs[c]
                        - classDataKNeighborRelation[c][lowerIndex]
                        - classDataKNeighborRelation[c][(int) upperIndex]
                        + cooccFreq) / (double) dataSize) * BasicMathUtil.log2(
                        ((double) (classFreqs[c] - classDataKNeighborRelation[
                        c][lowerIndex] - classDataKNeighborRelation[c][
                        (int) upperIndex] + cooccFreq + laplaceEstimatorSmall)
                        / ((double) classFreqs[c] + laplaceEstimatorSmall))
                        / ((1 - ((classDataKNeighborRelation[c][
                        (int) upperIndex] + laplaceEstimatorSmall) / (
                        (double) classFreqs[c] + laplaceEstimatorSmall))) * (1
                        - ((classDataKNeighborRelation[c][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[c]
                        + laplaceEstimatorSmall)))));
            }
            mutualInformationMap.put(concat, bothOccurFactor
                    + firstOccursFactor + secondOccursFactor +
                    noneOccursFactor);
        }
        // Normalize class-to-class priors.
        double laplaceTotal;
        laplaceEstimatorSmall = 0.00001f;
        laplaceTotal = numClasses * laplaceEstimatorSmall;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimatorSmall;
                classToClassPriors[cFirst][cSecond] /= ((k + 1)
                        * classFreqs[cSecond] * classFreqs[cFirst]
                        + laplaceTotal);
            }
        }
        // Now we calculate how often classes co-occur in neighborhoods of each
        // particular class.
        float occSum;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                occSum = 0;
                for (int cThird = 0; cThird < numClasses; cThird++) {
                    occSum += classCoOccurrencesInNeighborhoodsOfClasses[
                            cFirst][cSecond][cThird];
                }
                if (occSum > 0) {
                    for (int cThird = 0; cThird < numClasses; cThird++) {
                        classCoOccurrencesInNeighborhoodsOfClasses[cFirst][
                                cSecond][cThird] /= occSum;
                    }
                }
            }
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
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
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        // Find the k-nearest neighbors.
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
        for (int i = 0; i < numClasses; i++) {
            classProbEstimates[i] = classPriors[i];
        }
        // ODE[i][j] quantifies how i is conditioned on j.
        double[][][] ODEs = new double[numClasses][k][k];
        // weights[i][j] quantifies the strength of how i is conditioned on j.
        double[][][] weights = new double[numClasses][k][k];
        int lowerIndex;
        long upperIndex;
        long concat;
        for (int kIndexFirst = 0; kIndexFirst < k; kIndexFirst++) {
            for (int kIndexSecond = kIndexFirst + 1; kIndexSecond < k;
                    kIndexSecond++) {
                // Encode the neighbor pair.
                lowerIndex = Math.min(kNeighbors[kIndexFirst],
                        kNeighbors[kIndexSecond]);
                upperIndex = Math.max(kNeighbors[kIndexFirst],
                        kNeighbors[kIndexSecond]);
                concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                if (neighbOccFreqs[kNeighbors[kIndexFirst]] > thetaValue) {
                    // A regular point or a hub point, above the anti-hub
                    // threshold.
                    for (int c = 0; c < numClasses; c++) {
                        if (coDependencyMaps[c].containsKey(concat)) {
                            // These neighbors have co-occurred on the training
                            // data in neighborhoods of class c.
                            ODEs[c][kIndexSecond][kIndexFirst] =
                                    ((double) coDependencyMaps[c].get(concat)
                                    + laplaceEstimatorSmall)
                                    / ((double) classDataKNeighborRelation[c][
                                    kNeighbors[kIndexFirst]]
                                    + laplaceEstimatorSmall);
                            if (classConditionalSelfInformation[kIndexFirst][c]
                                    != 0) {
                                // Calculate the interaction strength.
                                weights[c][kIndexSecond][kIndexFirst] =
                                        mutualInformationMap.get(concat)
                                        / classConditionalSelfInformation[
                                        kIndexFirst][c];
                                weights[c][kIndexSecond][kIndexFirst] /=
                                        (rnnImpurity[kIndexFirst]
                                        + laplaceEstimatorBig);
                            } else {
                                weights[c][kIndexSecond][kIndexFirst] =
                                        mutualInformationMap.get(concat)
                                        / BasicMathUtil.log2(
                                        dataSize);
                            }
                        } else {
                            // These neighbors have not co-occurred before in
                            // neighborhoods of the current class c.
                            if (classDataKNeighborRelation[c][
                                    kNeighbors[kIndexFirst]] > 0) {
                                ODEs[c][kIndexSecond][kIndexFirst] =
                                        (laplaceEstimatorSmall)
                                        / ((double) classDataKNeighborRelation[
                                        c][kNeighbors[kIndexFirst]]
                                        + laplaceEstimatorSmall);
                                if (classConditionalSelfInformation[
                                        kIndexFirst][c] != 0) {
                                    weights[c][kIndexSecond][kIndexFirst] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / classConditionalSelfInformation[
                                            kIndexFirst][c];
                                    weights[c][kIndexSecond][kIndexFirst] /=
                                            (rnnImpurity[kIndexFirst]
                                            + laplaceEstimatorBig);
                                } else {
                                    weights[c][kIndexSecond][kIndexFirst] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / BasicMathUtil.log2(
                                            dataSize);
                                }
                            } else {
                                ODEs[c][kIndexSecond][kIndexFirst] =
                                        classCoOccurrencesInNeighborhoodsOfClasses[
                                        c][trainingData.getLabelOf(
                                        kNeighbors[kIndexFirst])][
                                        trainingData.getLabelOf(
                                        kNeighbors[kIndexSecond])];
                                weights[c][kIndexSecond][kIndexFirst] =
                                        (calculateMutualInformation(
                                        lowerIndex, upperIndex)
                                        + laplaceEstimatorSmall)
                                        / BasicMathUtil.log2(dataSize
                                        / k);
                            }
                        }
                    }
                } else {
                    // The special anti-hub handling case.
                    for (int c = 0; c < numClasses; c++) {
                        ODEs[c][kIndexSecond][kIndexFirst] = 0;
                        weights[c][kIndexSecond][kIndexFirst] = 0;
                    }
                }
                // Now for the second neighbor in comparison.
                if (neighbOccFreqs[kNeighbors[kIndexSecond]] > thetaValue) {
                    for (int c = 0; c < numClasses; c++) {
                        if (coDependencyMaps[c].containsKey(concat)) {
                            // These neighbors have co-occurred on the training
                            // data in neighborhoods of class c.
                            ODEs[c][kIndexFirst][kIndexSecond] = (
                                    (double) coDependencyMaps[c].get(concat)
                                    + laplaceEstimatorSmall) /
                                    ((double) classDataKNeighborRelation[c][
                                    kNeighbors[kIndexSecond]] +
                                    laplaceEstimatorSmall);
                            // Calculate the interaction strength.
                            if (classConditionalSelfInformation[kIndexSecond][c]
                                    != 0) {
                                weights[c][kIndexFirst][kIndexSecond] =
                                        mutualInformationMap.get(concat)
                                        / classConditionalSelfInformation[
                                        kIndexSecond][c];
                                weights[c][kIndexSecond][kIndexFirst] /=
                                        (rnnImpurity[kIndexSecond]
                                        + laplaceEstimatorBig);
                            } else {
                                weights[c][kIndexFirst][kIndexSecond] =
                                        mutualInformationMap.get(concat)
                                        / BasicMathUtil.log2(dataSize);
                            }
                        } else {
                            // These neighbors have not co-occurred before in
                            // neighborhoods of class c.
                            if (classDataKNeighborRelation[c][kNeighbors[
                                    kIndexSecond]] > 0) {
                                ODEs[c][kIndexFirst][kIndexSecond] =
                                        (laplaceEstimatorSmall) /
                                        ((double) classDataKNeighborRelation[c][
                                        kNeighbors[kIndexSecond]]
                                        + laplaceEstimatorSmall);
                                if (classConditionalSelfInformation[
                                        kIndexSecond][c] != 0) {
                                    weights[c][kIndexFirst][kIndexSecond] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / classConditionalSelfInformation[
                                            kIndexSecond][c];
                                    weights[c][kIndexSecond][kIndexFirst] /=
                                            (rnnImpurity[kIndexSecond]
                                            + laplaceEstimatorBig);
                                } else {
                                    weights[c][kIndexFirst][kIndexSecond] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / BasicMathUtil.log2(
                                            dataSize);
                                }
                            } else {
                                ODEs[c][kIndexFirst][kIndexSecond] =
                                        classCoOccurrencesInNeighborhoodsOfClasses[
                                        c][trainingData.getLabelOf(kNeighbors[
                                        kIndexSecond])][trainingData.getLabelOf(
                                        kNeighbors[kIndexFirst])];
                                weights[c][kIndexFirst][kIndexSecond] =
                                        (calculateMutualInformation(lowerIndex,
                                        upperIndex) + laplaceEstimatorSmall)
                                        / BasicMathUtil.log2(dataSize
                                        / k);
                            }
                        }
                    }
                } else {
                    // The special anti-hub handling case.
                    for (int c = 0; c < numClasses; c++) {
                        ODEs[c][kIndexFirst][kIndexSecond] = 0;
                        weights[c][kIndexFirst][kIndexSecond] = 0;
                    }
                }
            }
        }
        // Normalize the weights for ODE combinations.
        double weightSum;
        for (int c = 0; c < numClasses; c++) {
            weightSum = 0;
            for (int kIndFirst = 0; kIndFirst < k; kIndFirst++) {
                for (int kIndSecond = 0; kIndSecond < k; kIndSecond++) {
                    weightSum += weights[c][kIndFirst][kIndSecond];
                }
                if (weightSum > 0) {
                    for (int kIndSecond = 0; kIndSecond < k; kIndSecond++) {
                        weights[c][kIndFirst][kIndSecond] /= weightSum;
                    }
                }
            }
        }
        // Combine all the interactions.
        double multiplicativeFactor;
        double maxProb = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighbOccFreqs[kNeighbors[kIndex]] > thetaValue) {
                for (int c = 0; c < numClasses; c++) {
                    multiplicativeFactor = 0;
                    for (int j = 0; j < k; j++) {
                        multiplicativeFactor += weights[c][kIndex][j]
                                * ODEs[c][kIndex][j];
                    }
                    classProbEstimates[c] *= multiplicativeFactor;
                    if (classProbEstimates[c] > maxProb) {
                        maxProb = classProbEstimates[c];
                    }
                }
            } else {
                for (int c = 0; c < numClasses; c++) {
                    multiplicativeFactor = (double) classToClassPriors[
                            trainingData.getLabelOf(kNeighbors[kIndex])][c];
                    classProbEstimates[c] *= multiplicativeFactor;
                    if (classProbEstimates[c] > maxProb) {
                        maxProb = classProbEstimates[c];
                    }
                }
            }
            if (maxProb > 0) {
                for (int c = 0; c < numClasses; c++) {
                    classProbEstimates[c] /= maxProb;
                }
            }
        }
        // Normalize the probabilities.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        float[] cProbEstimatesFloat = new float[numClasses];
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
                cProbEstimatesFloat[cIndex] = (float) classProbEstimates[
                        cIndex];
            }
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] = classPriors[cIndex];
                cProbEstimatesFloat[cIndex] = (float) classProbEstimates[
                        cIndex];
            }
        }
        return cProbEstimatesFloat;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // First find the k-nearest neighbors.
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
        for (int i = 0; i < numClasses; i++) {
            classProbEstimates[i] = classPriors[i];
        }
        // ODE[i][j] quantifies how i is conditioned on j.
        double[][][] ODEs = new double[numClasses][k][k];
        // weights[i][j] quantifies the strength of how i is conditioned on j.
        double[][][] weights = new double[numClasses][k][k];
        int lowerIndex;
        long upperIndex;
        long concat;
        for (int kIndexFirst = 0; kIndexFirst < k; kIndexFirst++) {
            for (int kIndexSecond = kIndexFirst + 1; kIndexSecond < k;
                    kIndexSecond++) {
                // Encode the neighbor pair.
                lowerIndex = Math.min(kNeighbors[kIndexFirst],
                        kNeighbors[kIndexSecond]);
                upperIndex = Math.max(kNeighbors[kIndexFirst],
                        kNeighbors[kIndexSecond]);
                concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                if (neighbOccFreqs[kNeighbors[kIndexFirst]] > thetaValue) {
                    // A regular point or a hub point, above the anti-hub
                    // threshold.
                    for (int c = 0; c < numClasses; c++) {
                        if (coDependencyMaps[c].containsKey(concat)) {
                            // These neighbors have co-occurred on the training
                            // data in neighborhoods of class c.
                            ODEs[c][kIndexSecond][kIndexFirst] =
                                    ((double) coDependencyMaps[c].get(concat)
                                    + laplaceEstimatorSmall) /
                                    ((double) classDataKNeighborRelation[c][
                                    kNeighbors[kIndexFirst]] +
                                    laplaceEstimatorSmall);
                            // Calculate the interaction strength.
                            if (classConditionalSelfInformation[kIndexFirst][c]
                                    != 0) {
                                weights[c][kIndexSecond][kIndexFirst] =
                                        mutualInformationMap.get(concat)
                                        / classConditionalSelfInformation[
                                        kIndexFirst][c];
                                weights[c][kIndexSecond][kIndexFirst] /=
                                        (rnnImpurity[kIndexFirst]
                                        + laplaceEstimatorBig);
                            } else {
                                weights[c][kIndexSecond][kIndexFirst] =
                                        mutualInformationMap.get(concat)
                                        / BasicMathUtil.log2(dataSize);
                            }
                        } else {
                            // These neighbors have not co-occurred before in
                            // neighborhoods of the current class c.
                            if (classDataKNeighborRelation[c][kNeighbors[
                                    kIndexFirst]] > 0) {
                                ODEs[c][kIndexSecond][kIndexFirst] = (
                                        laplaceEstimatorSmall) /
                                        ((double) classDataKNeighborRelation[c][
                                        kNeighbors[kIndexFirst]]
                                        + laplaceEstimatorSmall);
                                if (classConditionalSelfInformation[
                                        kIndexFirst][c] != 0) {
                                    weights[c][kIndexSecond][kIndexFirst] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / classConditionalSelfInformation[
                                            kIndexFirst][c];
                                    weights[c][kIndexSecond][kIndexFirst] /=
                                            (rnnImpurity[kIndexFirst]
                                            + laplaceEstimatorBig);
                                } else {
                                    weights[c][kIndexSecond][kIndexFirst] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / BasicMathUtil.log2(
                                            dataSize);
                                }
                            } else {
                                ODEs[c][kIndexSecond][kIndexFirst] =
                                        classCoOccurrencesInNeighborhoodsOfClasses[
                                        c][trainingData.getLabelOf(
                                        kNeighbors[kIndexFirst])][
                                        trainingData.getLabelOf(
                                        kNeighbors[kIndexSecond])];
                                weights[c][kIndexSecond][kIndexFirst] =
                                        (calculateMutualInformation(lowerIndex,
                                        upperIndex) + laplaceEstimatorSmall)
                                        / BasicMathUtil.log2(dataSize / k);
                            }
                        }
                    }
                } else {
                    // The special anti-hub handling case.
                    for (int c = 0; c < numClasses; c++) {
                        ODEs[c][kIndexSecond][kIndexFirst] = 0;
                        weights[c][kIndexSecond][kIndexFirst] = 0;
                    }
                }
                // Now for the second neighbor in comparison.
                if (neighbOccFreqs[kNeighbors[kIndexSecond]] > thetaValue) {
                    for (int c = 0; c < numClasses; c++) {
                        if (coDependencyMaps[c].containsKey(concat)) {
                            // These neighbors have co-occurred on the training
                            // data in neighborhoods of class c.
                            ODEs[c][kIndexFirst][kIndexSecond] =
                                    ((double) coDependencyMaps[c].get(concat)
                                    + laplaceEstimatorSmall) /
                                    ((double) classDataKNeighborRelation[c][
                                    kNeighbors[kIndexSecond]] +
                                    laplaceEstimatorSmall);
                            // Calculate the interaction strength.
                            if (classConditionalSelfInformation[kIndexSecond][c]
                                    != 0) {
                                weights[c][kIndexFirst][kIndexSecond] =
                                        mutualInformationMap.get(concat)
                                        / classConditionalSelfInformation[
                                        kIndexSecond][c];
                                weights[c][kIndexSecond][kIndexFirst] /=
                                        (rnnImpurity[kIndexSecond]
                                        + laplaceEstimatorBig);
                            } else {
                                weights[c][kIndexFirst][kIndexSecond] =
                                        mutualInformationMap.get(concat)
                                        / BasicMathUtil.log2(dataSize);
                            }
                        } else {
                            // These neighbors have not co-occurred before in
                            // neighborhoods of class c.
                            if (classDataKNeighborRelation[c][kNeighbors[
                                    kIndexSecond]] > 0) {
                                ODEs[c][kIndexFirst][kIndexSecond] =
                                        (laplaceEstimatorSmall) /
                                        ((double) classDataKNeighborRelation[c][
                                        kNeighbors[kIndexSecond]]
                                        + laplaceEstimatorSmall);
                                if (classConditionalSelfInformation[
                                        kIndexSecond][c] != 0) {
                                    weights[c][kIndexFirst][kIndexSecond] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / classConditionalSelfInformation[
                                            kIndexSecond][c];
                                    weights[c][kIndexSecond][kIndexFirst] /=
                                            (rnnImpurity[kIndexSecond]
                                            + laplaceEstimatorBig);
                                } else {
                                    weights[c][kIndexFirst][kIndexSecond] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / BasicMathUtil.log2(dataSize);
                                }
                            } else {
                                ODEs[c][kIndexFirst][kIndexSecond] = 0;
                                weights[c][kIndexFirst][kIndexSecond] = 0;
                            }
                        }
                    }
                } else {
                    // The special anti-hub handling case.
                    for (int c = 0; c < numClasses; c++) {
                        ODEs[c][kIndexFirst][kIndexSecond] =
                                classCoOccurrencesInNeighborhoodsOfClasses[c][
                                trainingData.getLabelOf(kNeighbors[
                                kIndexSecond])][trainingData.getLabelOf(
                                kNeighbors[kIndexFirst])];
                        weights[c][kIndexFirst][kIndexSecond] =
                                (calculateMutualInformation(lowerIndex,
                                upperIndex) + laplaceEstimatorSmall)
                                / BasicMathUtil.log2(dataSize / k);
                    }
                }
            }
        }
        // Normalize the weights for ODE combinations.
        double weightSum;
        for (int c = 0; c < numClasses; c++) {
            weightSum = 0;
            for (int kIndFirst = 0; kIndFirst < k; kIndFirst++) {
                for (int kIndSecond = 0; kIndSecond < k; kIndSecond++) {
                    weightSum += weights[c][kIndFirst][kIndSecond];
                }
                if (weightSum > 0) {
                    for (int kIndSecond = 0; kIndSecond < k; kIndSecond++) {
                        weights[c][kIndFirst][kIndSecond] /= weightSum;
                    }
                }
            }
        }
        // Combine all the interactions.
        double multiplicativeFactor;
        double maxProb = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighbOccFreqs[kNeighbors[kIndex]] > thetaValue) {
                for (int c = 0; c < numClasses; c++) {
                    multiplicativeFactor = 0;
                    for (int j = 0; j < k; j++) {
                        multiplicativeFactor += weights[c][kIndex][j]
                                * ODEs[c][kIndex][j];
                    }
                    classProbEstimates[c] *= multiplicativeFactor;
                    if (classProbEstimates[c] > maxProb) {
                        maxProb = classProbEstimates[c];
                    }
                }
            } else {
                for (int c = 0; c < numClasses; c++) {
                    multiplicativeFactor = (double) classToClassPriors[
                            trainingData.getLabelOf(kNeighbors[kIndex])][c];
                    classProbEstimates[c] *= multiplicativeFactor;
                    if (classProbEstimates[c] > maxProb) {
                        maxProb = classProbEstimates[c];
                    }
                }
            }
            if (maxProb > 0) {
                for (int c = 0; c < numClasses; c++) {
                    classProbEstimates[c] /= maxProb;
                }
            }
        }
        // Normalize the probabilities.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        float[] cProbEstimatesFloat = new float[numClasses];
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
                cProbEstimatesFloat[cIndex] = (float) classProbEstimates[
                        cIndex];
            }
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] = classPriors[cIndex];
                cProbEstimatesFloat[cIndex] = (float) classProbEstimates[
                        cIndex];
            }
        }
        return cProbEstimatesFloat;
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
        for (int i = 0; i < numClasses; i++) {
            classProbEstimates[i] = classPriors[i];
        }
        // ODE[i][j] quantifies how i is conditioned on j.
        double[][][] ODEs = new double[numClasses][k][k];
        // weights[i][j] quantifies the strength of how i is conditioned on j.
        double[][][] weights = new double[numClasses][k][k];
        int lowerIndex;
        long upperIndex;
        long concat;
        for (int kIndFirst = 0; kIndFirst < k; kIndFirst++) {
            for (int kIndSecond = kIndFirst + 1; kIndSecond < k; kIndSecond++) {
                // Encode the neighbor pair.
                lowerIndex = Math.min(trNeighbors[kIndFirst],
                        trNeighbors[kIndSecond]);
                upperIndex = Math.max(trNeighbors[kIndFirst],
                        trNeighbors[kIndSecond]);
                concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
                if (neighbOccFreqs[trNeighbors[kIndFirst]] > thetaValue) {
                    // A regular point or a hub point, above the anti-hub
                    // threshold.
                    for (int c = 0; c < numClasses; c++) {
                        if (coDependencyMaps[c].containsKey(concat)) {
                            // These neighbors have co-occurred on the training
                            // data in neighborhoods of class c.
                            ODEs[c][kIndSecond][kIndFirst] =
                                    ((double) coDependencyMaps[c].get(concat)
                                    + laplaceEstimatorSmall) /
                                    ((double) classDataKNeighborRelation[c][
                                    trNeighbors[kIndFirst]] +
                                    laplaceEstimatorSmall);
                            // Calculate the interaction strength.
                            if (classConditionalSelfInformation[kIndFirst][c]
                                    != 0) {
                                weights[c][kIndSecond][kIndFirst] =
                                        (mutualInformationMap.get(concat)
                                        + laplaceEstimatorSmall)
                                        / classConditionalSelfInformation[
                                        kIndFirst][c];
                                weights[c][kIndSecond][kIndFirst] /=
                                        (rnnImpurity[kIndFirst]
                                        + laplaceEstimatorBig);
                            } else {
                                weights[c][kIndSecond][kIndFirst] =
                                        (mutualInformationMap.get(concat)
                                        + laplaceEstimatorSmall)
                                        / BasicMathUtil.log2(dataSize);
                            }
                        } else {
                            // These neighbors have not co-occurred before in
                            // neighborhoods of the current class c.
                            if (classDataKNeighborRelation[c][
                                    trNeighbors[kIndFirst]] > 0) {
                                ODEs[c][kIndSecond][kIndFirst] = (
                                        laplaceEstimatorSmall) /
                                        ((double) classDataKNeighborRelation[c][
                                        trNeighbors[kIndFirst]]
                                        + laplaceEstimatorSmall);
                                if (classConditionalSelfInformation[kIndFirst][
                                        c] != 0) {
                                    weights[c][kIndSecond][kIndFirst] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / classConditionalSelfInformation[
                                            kIndFirst][c];
                                    weights[c][kIndSecond][kIndFirst] /=
                                            (rnnImpurity[kIndFirst]
                                            + laplaceEstimatorBig);
                                } else {
                                    weights[c][kIndSecond][kIndFirst] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / BasicMathUtil.log2(dataSize);
                                }
                            } else {
                                ODEs[c][kIndSecond][kIndFirst] = 0;
                                weights[c][kIndSecond][kIndFirst] = 0;
                            }
                        }
                    }
                } else {
                    // The special anti-hub handling case.
                    for (int c = 0; c < numClasses; c++) {
                        ODEs[c][kIndSecond][kIndFirst] = 0;
                        weights[c][kIndSecond][kIndFirst] = 0;
                    }
                }
                // Now for the second neighbor in comparison.
                if (neighbOccFreqs[trNeighbors[kIndSecond]] > thetaValue) {
                    // A regular point or a hub point, above the anti-hub
                    // threshold.
                    for (int c = 0; c < numClasses; c++) {
                        if (coDependencyMaps[c].containsKey(concat)) {
                            // These neighbors have co-occurred on the training
                            // data in neighborhoods of class c.
                            ODEs[c][kIndFirst][kIndSecond] =
                                    ((double) coDependencyMaps[c].get(concat)
                                    + laplaceEstimatorSmall) /
                                    ((double) classDataKNeighborRelation[c][
                                    trNeighbors[kIndSecond]] +
                                    laplaceEstimatorSmall);
                            // Calculate the interaction strength.
                            if (classConditionalSelfInformation[kIndSecond][c]
                                    != 0) {
                                weights[c][kIndFirst][kIndSecond] =
                                        (mutualInformationMap.get(concat)
                                        + laplaceEstimatorSmall)
                                        / classConditionalSelfInformation[
                                        kIndSecond][c];
                                weights[c][kIndSecond][kIndFirst] /=
                                        (rnnImpurity[kIndSecond]
                                        + laplaceEstimatorBig);
                            } else {
                                weights[c][kIndFirst][kIndSecond] =
                                        (mutualInformationMap.get(concat)
                                        + laplaceEstimatorSmall)
                                        / BasicMathUtil.log2(dataSize);
                            }
                        } else {
                            // These neighbors have not co-occurred before in
                            // neighborhoods of class c.
                            if (classDataKNeighborRelation[c][trNeighbors[
                                    kIndSecond]] > 0) {
                                ODEs[c][kIndFirst][kIndSecond] =
                                        (laplaceEstimatorSmall) /
                                        ((double) classDataKNeighborRelation[c][
                                        trNeighbors[kIndSecond]]
                                        + laplaceEstimatorSmall);
                                if (classConditionalSelfInformation[kIndSecond][
                                        c] != 0) {
                                    weights[c][kIndFirst][kIndSecond] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / classConditionalSelfInformation[
                                            kIndSecond][c];
                                    weights[c][kIndSecond][kIndFirst] /=
                                            (rnnImpurity[kIndSecond]
                                            + laplaceEstimatorBig);
                                } else {
                                    weights[c][kIndFirst][kIndSecond] =
                                            (calculateMutualInformation(
                                            lowerIndex, upperIndex)
                                            + laplaceEstimatorSmall)
                                            / BasicMathUtil.log2(dataSize);
                                }
                            } else {
                                ODEs[c][kIndFirst][kIndSecond] = 0;
                                weights[c][kIndFirst][kIndSecond] = 0;
                            }
                        }
                    }
                } else {
                    // The special anti-hub handling case.
                    for (int c = 0; c < numClasses; c++) {
                        ODEs[c][kIndFirst][kIndSecond] = 0;
                        weights[c][kIndFirst][kIndSecond] = 0;
                    }
                }
            }
        }
        // Normalize the weights for ODE combinations.
        double weightSum;
        for (int c = 0; c < numClasses; c++) {
            weightSum = 0;
            for (int kIndFirst = 0; kIndFirst < k; kIndFirst++) {
                for (int kIndSecond = 0; kIndSecond < k; kIndSecond++) {
                    weightSum += weights[c][kIndFirst][kIndSecond];
                }
                if (weightSum > 0) {
                    for (int kIndSecond = 0; kIndSecond < k; kIndSecond++) {
                        weights[c][kIndFirst][kIndSecond] /= weightSum;
                    }
                }
            }
        }
        // Combine all the interactions.
        double multiplicativeFactor;
        double maxProb = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighbOccFreqs[trNeighbors[kIndex]] > thetaValue) {
                for (int c = 0; c < numClasses; c++) {
                    multiplicativeFactor = 0;
                    for (int j = 0; j < k; j++) {
                        multiplicativeFactor += weights[c][kIndex][j]
                                * ODEs[c][kIndex][j];
                    }
                    classProbEstimates[c] *= multiplicativeFactor;
                    if (classProbEstimates[c] > maxProb) {
                        maxProb = classProbEstimates[c];
                    }
                }
            } else {
                for (int c = 0; c < numClasses; c++) {
                    multiplicativeFactor = (double) classToClassPriors[
                            trainingData.getLabelOf(trNeighbors[kIndex])][c];
                    classProbEstimates[c] *= multiplicativeFactor;
                    if (classProbEstimates[c] > maxProb) {
                        maxProb = classProbEstimates[c];
                    }
                }
            }
            if (maxProb > 0) {
                for (int c = 0; c < numClasses; c++) {
                    classProbEstimates[c] /= maxProb;
                }
            }
        }
        // Normalize the probabilities.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        float[] cProbEstimatesFloat = new float[numClasses];
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
                cProbEstimatesFloat[cIndex] = (float) classProbEstimates[
                        cIndex];
            }
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] = classPriors[cIndex];
                cProbEstimatesFloat[cIndex] = (float) classProbEstimates[
                        cIndex];
            }
        }
        return cProbEstimatesFloat;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
