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

import algref.Address;
import algref.Author;
import algref.JournalPublication;
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
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;
import util.AuxSort;
import util.BasicMathUtil;

/**
 * This class implements the HIKNN algorithm that was proposed in the paper
 * titled: "Nearest Neighbor Voting in High Dimensional Data: Learning from Past
 * Occurrences" published in Computer Science and Information Systems in 2011.
 * The algorithm is an extension of h-FNN that gives preference to rare neighbor
 * points and uses some label information.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HIKNN extends Classifier implements AutomaticKFinderInterface,
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {

    private static final long serialVersionUID = 1L;
    private int k = 5;
    // NeighborSetFinder object for kNN calculations.
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    // The distance weighting parameter.
    private float mValue = 2;
    private float[][] classDataKNeighborRelation = null;
    // Information contained in the neighbors' labels.
    private float[] labelInformationFactor = null;
    // The prior class distribution.
    private float[] classPriors = null;
    private float laplaceEstimator = 0.001f;
    private int[] neighborOccurrenceFreqs = null;
    // The distance matrix.
    private float[][] distMat;
    private int dataSize;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Nearest Neighbor Voting in High-Dimensional Data: "
                + "Learning from Past Occurrences");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.setPublisher(new Publisher("ComSIS Consortium",
                new Address("Novi Sad", "Serbia")));
        pub.setJournalName("Computer Science and Information Systems");
        pub.setYear(2013);
        pub.setStartPage(691);
        pub.setEndPage(712);
        pub.setVolume(9);
        pub.setIssue(2);
        pub.setDoi("10.2298/CSIS111211014T");
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "HIKNN";
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }
    boolean noRecalc = false;

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    /**
     * Finds the optimal k-value on the training data by a leave-one-out
     * procedure.
     *
     * @param kMin Integer that is the minimal k-value.
     * @param kMax Integer that is the maximal k-value.
     * @throws Exception
     */
    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DataSet dset = trainingData;
        NeighborSetFinder nsfLOU;
        if (distMat == null) {
            nsfLOU = new NeighborSetFinder(dset, getCombinedMetric());
            nsfLOU.calculateDistances();
        } else {
            nsfLOU = new NeighborSetFinder(dset, distMat,
                    getCombinedMetric());
        }
        nsfLOU.calculateNeighborSets(kMax + 1);
        nsfLOU.recalculateStatsForSmallerK(kMax);
        // This will hold the accuracy of each k-choice.
        float[] accuracyArray = new float[kMax - kMin + 1];
        float currMaxAcc = -1f;
        int currMaxK = 0;
        // The optimal M-value.
        float currMaxM = 2;
        int numElements = dset.size();
        float currMaxVote;
        // The kNN sets.
        int[][] kneighbors = nsfLOU.getKNeighbors();
        float currAccuracy;

        float[][] kDistances = nsfLOU.getKDistances();
        // Trying 10 different M values, from 1 to 2 by 0.2.
        float[][][] distance_weightsAll = new float[numElements][kMax][11];
        float[][] dwSumAll = new float[numElements][11];
        // First generate the distance weights.
        for (int index = 0; index < numElements; index++) {
            for (int i = 0; i < kMax; i++) {
                mValue = 0.8f;
                for (int mIndex = 0; mIndex < 11; mIndex++) {
                    mValue += 0.2;
                    if (kDistances[index][i] != 0) {
                        distance_weightsAll[index][i][mIndex] = 1f
                                / ((float) Math.pow(kDistances[index][i],
                                (2f / (mValue - 1f))));
                    } else {
                        distance_weightsAll[index][i][mIndex] = 10000f;
                    }
                    dwSumAll[index][mIndex] +=
                            distance_weightsAll[index][i][mIndex];
                }
            }
        }
        classPriors = dset.getClassPriors();
        int currClass;
        // Get the class-conditional neighbor occurrence frequencies.
        float[][][] classDataKNeighborRelationAllK =
                new float[accuracyArray.length][numClasses][
                        trainingData.size()];
        int[][] neighbOccFreqsAllK = new int[kMax][];
        float[][] labelInformationAllK =
                new float[accuracyArray.length][trainingData.size()];
        ArrayList<Integer>[][] reverseNeighborsAllK =
                new ArrayList[accuracyArray.length][trainingData.size()];
        // The informativeness of the occurrence.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        // For all the neighborhood sizes.
        for (int kCurr = kMin; kCurr <= kMax; kCurr++) {
            nsfLOU.recalculateStatsForSmallerK(kCurr);
            neighbOccFreqsAllK[kCurr - 1] =
                    Arrays.copyOf(nsfLOU.getNeighborFrequencies(),
                    nsfLOU.getNeighborFrequencies().length);
            for (int i = 0; i < trainingData.size(); i++) {
                // Reverse neighbor lists are used for a quick leave-one-out.
                reverseNeighborsAllK[kCurr - kMin][i] =
                        (ArrayList<Integer>) (
                        nsfLOU.getReverseNeighbors()[i].clone());
                // Ascending sort.
                AuxSort.sortIntArrayList(
                        reverseNeighborsAllK[kCurr - kMin][i], false);
            }
            float maxHubness = 0;
            float minHubness = Float.MAX_VALUE;
            // Get the maximum and minimum neighbor occurrence frquency.
            for (int i = 0; i < trainingData.size(); i++) {
                if (neighbOccFreqsAllK[kCurr - 1][i] > maxHubness) {
                    maxHubness = neighbOccFreqsAllK[kCurr - 1][i];
                }
                if (neighbOccFreqsAllK[kCurr - 1][i] < minHubness) {
                    minHubness = neighbOccFreqsAllK[kCurr - 1][i];
                }
            }
            minEventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size()) / (maxHubness + 1f));
            maxEventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size()) / (0 + 1f));
            for (int i = 0; i < trainingData.size(); i++) {
                // Calculate the occurrence informativeness and the label
                // contribution factor.
                eventInfo = (float) BasicMathUtil.log2(
                        ((float) trainingData.size())
                        / ((float) neighbOccFreqsAllK[kCurr - 1][i] + 1f));
                labelInformationAllK[kCurr - kMin][i] =
                        (eventInfo - minEventInfo)
                        / (maxEventInfo - minEventInfo + 0.0001f);
                currClass = trainingData.data.get(i).getCategory();
                classDataKNeighborRelationAllK[kCurr - kMin][currClass][i]++;
                for (int j = 0; j < kCurr; j++) {
                    classDataKNeighborRelationAllK[kCurr - kMin][currClass][
                            kneighbors[i][j]]++;
                }
            }
        }
        // We do random flips for equal vote sums.
        Random randa = new Random();
        for (int mIndex = 0; mIndex < 11; mIndex++) {
            // For all the k-values.
            for (int i = 0; i < accuracyArray.length; i++) {
                float[][] currVoteCounts = new float[numElements][numClasses];
                int[] currVoteLabels = new int[numElements];
                // Find the accuracy for all the parameter options.
                currAccuracy = 0;
                for (int index = 0; index < numElements; index++) {
                    // Remove and replace, do a leave-one-out.
                    ArrayList<Integer>[] replacements =
                            new ArrayList[accuracyArray.length];
                    for (int i1 = 0; i1 < accuracyArray.length; i1++) {
                        replacements[i1] =
                                new ArrayList<>(neighbOccFreqsAllK[i1][index]);
                        for (int j = 0; j
                                < reverseNeighborsAllK[i1][index].size(); j++) {
                            replacements[i1].add(kneighbors[
                                    reverseNeighborsAllK[i1][index].get(j)][
                                    kMin + i1]);
                            classDataKNeighborRelationAllK[i1][
                                    trainingData.data.get(
                                    reverseNeighborsAllK[i1][index].get(j)).
                                    getCategory()][kneighbors[
                                    reverseNeighborsAllK[i1][index].get(j)][
                                    kMin + i1]]++;
                            neighbOccFreqsAllK[kMin + i1 - 1][
                                    kneighbors[reverseNeighborsAllK[i1][
                                    index].get(j)][kMin + i1]]++;
                        }
                    }
                    // Tracks the current maximum vote.
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        for (int kCurr = 0; kCurr <= kMin + i - 1; kCurr++) {
                            if (trainingData.data.get(kneighbors[index][kCurr]).
                                    getCategory() == classIndex) {
                                // Perform the HIKNN vote.
                                currVoteCounts[index][classIndex] +=
                                        ((labelInformationAllK[i][kneighbors[
                                        index][kCurr]] + (1
                                        - labelInformationAllK[i][
                                        kneighbors[index][kCurr]]) *
                                        ((classDataKNeighborRelationAllK[i][
                                        classIndex][kneighbors[index][kCurr]])
                                        / (float) (neighbOccFreqsAllK[
                                        kMin + i - 1][
                                        kneighbors[index][kCurr]] + 1f)))
                                        * (float) BasicMathUtil.log2(
                                        ((float) trainingData.size()) / (1f
                                        + neighbOccFreqsAllK[kMin + i - 1][
                                        kneighbors[index][kCurr]])))
                                        * distance_weightsAll[index][
                                        kMin + i - 1][mIndex] / dwSumAll[
                                        index][mIndex];
                            } else {
                                // Perform the HIKNN vote.
                                currVoteCounts[index][classIndex] += (
                                        (1 - labelInformationAllK[i][
                                        kneighbors[index][kCurr]]) * (
                                        (classDataKNeighborRelationAllK[i][
                                        classIndex][kneighbors[index][kCurr]])
                                        / (float) (neighbOccFreqsAllK[
                                        kMin + i - 1][
                                        kneighbors[index][kCurr]] + 1f))
                                        * (float) BasicMathUtil.log2(
                                        ((float) trainingData.size()) / (1f
                                        + neighbOccFreqsAllK[kMin + i - 1][
                                        kneighbors[index][kCurr]])))
                                        * distance_weightsAll[index][
                                        kMin + i - 1][mIndex]
                                        / dwSumAll[index][mIndex];
                            }
                        }
                        if (currVoteCounts[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCounts[index][classIndex];
                            currVoteLabels[index] = classIndex;
                        } else if (currVoteCounts[index][classIndex]
                                == currMaxVote && randa.nextFloat() < 0.5f) {
                            // In case of a tie, randomly choose one option.
                            currMaxVote = currVoteCounts[index][classIndex];
                            currVoteLabels[index] = classIndex;
                        }
                    }
                    if (currVoteLabels[index] == dset.data.get(index).
                            getCategory()) {
                        // If the vote is correct, increase the current
                        // accuracy.
                        currAccuracy++;
                    }
                    // Fix the kNN stats after the leave-one-out to their
                    // previous values.
                    for (int i1 = 0; i1 < accuracyArray.length; i1++) {
                        for (int j = 0; j < reverseNeighborsAllK[i1][index].
                                size(); j++) {
                            classDataKNeighborRelationAllK[i1][
                                    trainingData.data.get(
                                    reverseNeighborsAllK[i1][index].get(j)).
                                    getCategory()][replacements[i1].get(j)]--;
                            neighbOccFreqsAllK[kMin + i1 - 1][
                                    replacements[i1].get(j)]--;
                        }
                    }

                }
                // Normalize the accuracy.
                currAccuracy /= (float) numElements;
                // Log the crrent accuracy.
                accuracyArray[i] = currAccuracy;
                // If it is the best accuracy so far, change the selected
                // parameter configuration.
                if (currMaxAcc < currAccuracy) {
                    currMaxAcc = currAccuracy;
                    currMaxK = kMin + i;
                    currMaxM = 1.0f + mIndex * 0.2f;
                }
            }
        }
        k = currMaxK;
        mValue = currMaxM;
    }

    /**
     * The default constructor.
     */
    public HIKNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public HIKNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNN(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     */
    public HIKNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     */
    public HIKNN(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNN(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     */
    public HIKNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
     * @param categories Category[] of classes to train on.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // As this is an internal DataSet object, data instances will not have
        // it set as their data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int j = 0; j < categories[cIndex].size(); j++) {
                categories[cIndex].getInstance(j).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(j));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
    }

    /**
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
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
        // An internal training data representation.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[
                indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames =
                categories[indexFirstNonEmptyClass]
                .getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames =
                categories[indexFirstNonEmptyClass].getInstance(0).
                getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int j = 0; j < categories[cIndex].size(); j++) {
                categories[cIndex].getInstance(j).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(j));
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
        HIKNN result = new HIKNN(k, laplaceEstimator,
                getCombinedMetric(), numClasses);
        result.noRecalc = noRecalc;
        return result;
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        // Obtain the corresponding index permutation in the evaluation
        // framework.
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(
                reducer.getPrototypeIndexes(), reducer.getOriginalDataSet());
        classPriors = trainingData.getClassPriors();
        // Get the unbiased prototype hubness estimate.
        int[] protoOccurrenceFrequencies = reducer.getPrototypeHubness();
        neighborOccurrenceFreqs = new int[protoOccurrenceFrequencies.length];
        for (int i = 0; i < neighborOccurrenceFreqs.length; i++) {
            neighborOccurrenceFreqs[i] =
                    protoOccurrenceFrequencies[indexPermutation.get(i)];
        }
        labelInformationFactor = new float[trainingData.size()];
        dataSize = reducer.getOriginalDataSize();
        float maxHubness = 0;
        float minHubness = Float.MAX_VALUE;
        for (int i = 0; i < trainingData.size(); i++) {
            if (neighborOccurrenceFreqs[i] > maxHubness) {
                maxHubness = neighborOccurrenceFreqs[i];
            }
            if (neighborOccurrenceFreqs[i] < minHubness) {
                minHubness = neighborOccurrenceFreqs[i];
            }
        }
        // Get the unbiased class-conditional occurrence frequencies.
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        float[][] classDataKNeighborRelationProto =
                reducer.getClassDataNeighborRelationforFuzzy(
                numClasses, laplaceEstimator);
        for (int c = 0; c < numClasses; c++) {
            for (int i = 0; i < neighborOccurrenceFreqs.length; i++) {
                classDataKNeighborRelation[c][i] =
                        classDataKNeighborRelationProto[c][
                        indexPermutation.get(i)];
            }
        }
        // Calculate the neighbor occurrence information content for all the
        // neighbors.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        minEventInfo = (float) BasicMathUtil.log2(
                ((float) dataSize) / (maxHubness + 1f));
        maxEventInfo = (float) BasicMathUtil.log2(
                ((float) dataSize) / (0 + 1f));
        for (int i = 0; i < trainingData.size(); i++) {
            eventInfo = (float) BasicMathUtil.log2(
                    ((float) dataSize)
                    / ((float) neighborOccurrenceFreqs[i] + 1f));
            labelInformationFactor[i] = (eventInfo - minEventInfo)
                    / (maxEventInfo - minEventInfo + 0.0001f);
        }
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // Setting an invalid k-value is a signal for automatically
            // searching for the optimal value in the low-value range.
            findK(1, 20);
        }
        if (nsf == null) {
            // If the kNN sets have not been provided, calculate them.
            calculateNeighborSets();
        }
        // Find the class priors.
        classPriors = trainingData.getClassPriors();
        dataSize = trainingData.size();
        if (!noRecalc) {
            nsf.recalculateStatsForSmallerK(k);
        }
        // Get the neighbor hubness.
        neighborOccurrenceFreqs = nsf.getNeighborFrequencies();
        // Find the class-conditional neighbor occurrence frequencies.
        int[][] kneighbors = nsf.getKNeighbors();
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        labelInformationFactor = new float[trainingData.size()];
        int currClass;
        float maxHubness = 0;
        float minHubness = Float.MAX_VALUE;
        for (int i = 0; i < trainingData.size(); i++) {
            if (neighborOccurrenceFreqs[i] > maxHubness) {
                maxHubness = neighborOccurrenceFreqs[i];
            }
            if (neighborOccurrenceFreqs[i] < minHubness) {
                minHubness = neighborOccurrenceFreqs[i];
            }
        }
        // Calculate the neighbor occurrence informativeness.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        minEventInfo = (float) BasicMathUtil.log2(
                ((float) trainingData.size()) / (maxHubness + 1f));
        maxEventInfo = (float) BasicMathUtil.log2(
                ((float) trainingData.size()) / (0 + 1f));
        for (int i = 0; i < trainingData.size(); i++) {
            eventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size())
                    / ((float) neighborOccurrenceFreqs[i] + 1f));
            labelInformationFactor[i] = (eventInfo - minEventInfo)
                    / (maxEventInfo - minEventInfo + 0.0001f);
            currClass = trainingData.data.get(i).getCategory();
            classDataKNeighborRelation[currClass][i]++;
            for (int j = 0; j < k; j++) {
                classDataKNeighborRelation[currClass][kneighbors[i][j]]++;
            }
        }
        // Normalization.
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            for (int j = 0; j < trainingData.size(); j++) {
                classDataKNeighborRelation[cIndex][j] /=
                        (neighborOccurrenceFreqs[j] + 1);
            }
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
        float maxProb = 0;
        int result = 0;
        for (int i = 0; i < numClasses; i++) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i];
                result = i;
            }
        }
        return result;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        // Find the k-nearest neighbors.
        float[] kDistances = new float[k];
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
        }
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
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int i = 0; i < k; i++) {
            if (kDistances[i] != 0) {
                distanceWeights[i] = 1f / ((float) Math.pow(kDistances[i],
                        (2f / (mValue - 1f))));
            } else {
                distanceWeights[i] = 10000f;
            }
            dwSum += distanceWeights[i];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (trainingData.data.get(kNeighbors[i]).getCategory() == j) {
                    classProbEstimates[j] +=
                            ((labelInformationFactor[kNeighbors[i]]
                            + ((1 - labelInformationFactor[kNeighbors[i]])
                            * classDataKNeighborRelation[j][kNeighbors[i]]))
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[kNeighbors[i]])))
                            * distanceWeights[i] / dwSum;
                } else {
                    classProbEstimates[j] +=
                            ((((1 - labelInformationFactor[kNeighbors[i]]))
                            * classDataKNeighborRelation[j][kNeighbors[i]])
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[kNeighbors[i]])))
                            * distanceWeights[i] / dwSum;
                }
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Find the k-nearest neighbors.
        float[] kDistances = new float[k];
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
        }
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
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (kDistances[kIndex] != 0) {
                distanceWeights[kIndex] = 1f / ((float) Math.pow(
                        kDistances[kIndex], (2f / (mValue - 1f))));
            } else {
                distanceWeights[kIndex] = 10000f;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (trainingData.data.get(kNeighbors[kIndex]).getCategory()
                        == cIndex) {
                    classProbEstimates[cIndex] +=
                            ((labelInformationFactor[kNeighbors[kIndex]]
                            + ((1 - labelInformationFactor[kNeighbors[kIndex]])
                            * classDataKNeighborRelation[cIndex][kNeighbors[
                            kIndex]])) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighborOccurrenceFreqs[
                            kNeighbors[kIndex]])))
                            * distanceWeights[kIndex] / dwSum;
                } else {
                    classProbEstimates[cIndex] += ((((1
                            - labelInformationFactor[kNeighbors[kIndex]]))
                            * classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]]) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighborOccurrenceFreqs[
                            kNeighbors[kIndex]])))
                            * distanceWeights[kIndex] / dwSum;
                }
            }

        }
        // Normalize the probabilities.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining);
        float maxProb = 0;
        int result = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                result = cIndex;
            }
        }
        return result;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining,
            int[] trNeighbors) throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining, trNeighbors);
        float maxProb = 0;
        int result = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                result = cIndex;
            }
        }
        return result;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
    float[] distToTraining, int[] trNeighbors) throws Exception {
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (distToTraining[trNeighbors[kIndex]] != 0) {
                distanceWeights[kIndex] = 1f / ((float) Math.pow(
                        distToTraining[trNeighbors[kIndex]], (2f
                        / (mValue - 1f))));
            } else {
                distanceWeights[kIndex] = 10000f;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (trainingData.data.get(trNeighbors[kIndex]).getCategory()
                        == cIndex) {
                    classProbEstimates[cIndex] +=
                            ((labelInformationFactor[trNeighbors[kIndex]]
                            + ((1 - labelInformationFactor[trNeighbors[kIndex]])
                            * classDataKNeighborRelation[cIndex][trNeighbors[
                            kIndex]])) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighborOccurrenceFreqs[
                            trNeighbors[kIndex]])))
                            * distanceWeights[kIndex] / dwSum;
                } else {
                    classProbEstimates[cIndex] +=
                            ((((1 - labelInformationFactor[
                            trNeighbors[kIndex]])) * classDataKNeighborRelation[
                            cIndex][trNeighbors[kIndex]])
                            * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighborOccurrenceFreqs[
                            trNeighbors[kIndex]])))
                            * distanceWeights[kIndex] / dwSum;
                }
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }

    /**
     * Classify a new instance.
     *
     * @param instance DataInstance object to classify.
     * @param kDists float[] of distances to the k-nearest neighbors.
     * @param trNeighbors int[] holding the indexes of the k-nearest neighbors.
     * @return float[] of class affiliation probabilities.
     * @throws Exception
     */
    public float[] classifyProbabilisticallyWithKDistAndNeighbors(
            DataInstance instance, float[] kDists, int[] trNeighbors)
            throws Exception {
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (kDists[kIndex] != 0) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(kDists[kIndex],
                        (2f / (mValue - 1f))));
            } else {
                distanceWeights[kIndex] = 10000f;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (trainingData.data.get(trNeighbors[kIndex]).getCategory()
                        == cIndex) {
                    classProbEstimates[cIndex] += (
                            (labelInformationFactor[trNeighbors[kIndex]]
                            + ((1 - labelInformationFactor[trNeighbors[kIndex]])
                            * classDataKNeighborRelation[cIndex][trNeighbors[
                            kIndex]])) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighborOccurrenceFreqs[
                            trNeighbors[kIndex]])))
                            * distanceWeights[kIndex] / dwSum;
                } else {
                    classProbEstimates[cIndex] += (
                            (((1 - labelInformationFactor[trNeighbors[kIndex]]))
                            * classDataKNeighborRelation[cIndex][
                            trNeighbors[kIndex]]) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighborOccurrenceFreqs[
                            trNeighbors[kIndex]])))
                            * distanceWeights[kIndex] / dwSum;
                }
            }
        }
        // Normalize the probabilities.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
