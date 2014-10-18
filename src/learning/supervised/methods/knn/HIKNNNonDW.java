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

import learning.supervised.interfaces.AutomaticKFinderInterface;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
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
 * points and uses some label information. This implementation doesn't use
 * distance weighting in voting.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HIKNNNonDW extends Classifier implements
        AutomaticKFinderInterface, DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private int k = 5;
    // NeighborSetFinder object for kNN calculations.
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    private float[][] classDataKNeighborRelation = null;
    // Information contained in the neighbors' labels.
    private float[] labelInformationFactor = null;
    private float[] classPriors = null;
    private float laplaceEstimator = 0.001f;
    private int[] neighbOccFreqs = null;
    // The distance matrix.
    private float[][] distMat;
    private boolean noRecalc = false;
    private int dataSize;

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
        return "HIKNN - no distance weights";
    }

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DataSet dset = trainingData;
        NeighborSetFinder nsfLOU;
        // Calculate the kNN sets.
        if (distMat == null) {
            nsfLOU = new NeighborSetFinder(dset, getCombinedMetric());
            nsfLOU.calculateDistances();
        } else {
            nsfLOU = new NeighborSetFinder(dset, distMat, getCombinedMetric());
        }
        nsfLOU.calculateNeighborSets(kMax + 1);
        nsfLOU.recalculateStatsForSmallerK(kMax);
        // The array that holds the accuracy achieved for each k-value.
        float[] accuracyArray = new float[kMax - kMin + 1];
        float currMaxAcc = -1f;
        int currMaxK = 0;
        int numElements = dset.size();
        float currMaxVote;
        int[][] kneighbors = nsfLOU.getKNeighbors();
        float currAccuracy;
        classPriors = trainingData.getClassPriors();
        int currClass;
        // Class-conditional neighbor occurrence frequencies for all k-values.
        float[][][] classDataKNeighborRelationAllK =
                new float[accuracyArray.length][numClasses][
                        trainingData.size()];
        int[][] neighbOccFreqsAllK = new int[kMax][];
        // The information value contained in the label.
        float[][] labelInformationAllK =
                new float[accuracyArray.length][trainingData.size()];
        // Lists of reverse neighbors.
        ArrayList<Integer>[][] reverseNeighborsAllK =
                new ArrayList[accuracyArray.length][trainingData.size()];
        // Neighbor occurrence informativeness.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        // Check for all k values in the range.
        for (int kCurr = kMin; kCurr <= kMax; kCurr++) {
            // Update the counts and the kNN sets.
            nsfLOU.recalculateStatsForSmallerK(kCurr);
            neighbOccFreqsAllK[kCurr - 1] = Arrays.copyOf(
                    nsfLOU.getNeighborFrequencies(),
                    nsfLOU.getNeighborFrequencies().length);
            for (int i = 0; i < trainingData.size(); i++) {
                reverseNeighborsAllK[kCurr - kMin][i] =
                        (ArrayList<Integer>) (
                        nsfLOU.getReverseNeighbors()[i].clone());
                // Ascending sort.
                AuxSort.sortIntArrayList(
                        reverseNeighborsAllK[kCurr - kMin][i], false);
            }
            // Find the maximum neighbor occurrence frequency.
            float maxHubness = 0;
            float minHubness = Float.MAX_VALUE;
            for (int i = 0; i < trainingData.size(); i++) {
                if (neighbOccFreqsAllK[kCurr - 1][i] > maxHubness) {
                    maxHubness = neighbOccFreqsAllK[kCurr - 1][i];
                }
                if (neighbOccFreqsAllK[kCurr - 1][i] < minHubness) {
                    minHubness = neighbOccFreqsAllK[kCurr - 1][i];
                }
            }
            // Calculate the maximum and minimum neighbor occurrence
            // informativeness, to be used for normalization.
            minEventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size()) / (maxHubness + 1f));
            maxEventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size()) / (0 + 1f));
            for (int i = 0; i < trainingData.size(); i++) {
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
        // Now simulate the voting and calculate the accuracy for each
        // neighborhood size.
        Random randa = new Random();
        for (int kInc = 0; kInc < accuracyArray.length; kInc++) {
            // Arrays for the estimation of the votes.
            float[][] currVoteCounts = new float[numElements][numClasses];
            int[] currVoteLabels = new int[numElements];
            currAccuracy = 0;
            for (int index = 0; index < numElements; index++) {
                // Remove the element from the lists to avoid over-fitting.
                ArrayList<Integer>[] replacements =
                        new ArrayList[accuracyArray.length];
                for (int kIncSecond = 0; kIncSecond < accuracyArray.length;
                        kIncSecond++) {
                    replacements[kIncSecond] = new ArrayList<>(
                            neighbOccFreqsAllK[kIncSecond][index]);
                    for (int j = 0; j < reverseNeighborsAllK[kIncSecond][
                            index].size(); j++) {
                        replacements[kIncSecond].add(
                                kneighbors[reverseNeighborsAllK[kIncSecond][
                                index].get(j)][kMin + kIncSecond]);
                        classDataKNeighborRelationAllK[kIncSecond][
                                trainingData.data.get(reverseNeighborsAllK[
                                kIncSecond][index].get(j)).getCategory()][
                                kneighbors[reverseNeighborsAllK[kIncSecond][
                                index].get(j)][kMin + kIncSecond]]++;
                        neighbOccFreqsAllK[kMin + kIncSecond - 1][
                                kneighbors[reverseNeighborsAllK[kIncSecond][
                                index].get(j)][kMin + kIncSecond]]++;
                    }
                }
                // Perform the voting.
                currMaxVote = 0;
                for (int classIndex = 0; classIndex < numClasses;
                        classIndex++) {
                    for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                        if (trainingData.data.get(kneighbors[index][kCurr]).
                                getCategory() == classIndex) {
                            // The combined vote.
                            currVoteCounts[index][classIndex] +=
                                    ((labelInformationAllK[kInc][
                                    kneighbors[index][kCurr]] + (1
                                    - labelInformationAllK[kInc][kneighbors[
                                    index][kCurr]])
                                    * ((classDataKNeighborRelationAllK[kInc][
                                    classIndex][kneighbors[index][kCurr]])
                                    / (float) (neighbOccFreqsAllK[
                                    kMin + kInc - 1][
                                    kneighbors[index][kCurr]] + 1f)))
                                    * (float) BasicMathUtil.log2(
                                    ((float) trainingData.size())
                                    / (1f + neighbOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]])));
                        } else {
                            // The pure class-conditional occurrence profile
                            // vote.
                            currVoteCounts[index][classIndex] +=
                                    ((1 - labelInformationAllK[kInc][
                                    kneighbors[index][kCurr]])
                                    * ((classDataKNeighborRelationAllK[kInc][
                                    classIndex][kneighbors[index][kCurr]])
                                    / (float) (neighbOccFreqsAllK[
                                    kMin + kInc - 1][
                                    kneighbors[index][kCurr]] + 1f))
                                    * (float) BasicMathUtil.log2(
                                    ((float) trainingData.size())
                                    / (1f + neighbOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]])));
                        }
                    }
                    // Update the decision.
                    if (currVoteCounts[index][classIndex] > currMaxVote) {
                        currMaxVote = currVoteCounts[index][classIndex];
                        currVoteLabels[index] = classIndex;
                    } else if (currVoteCounts[index][classIndex]
                            == currMaxVote && randa.nextFloat() < 0.5f) {
                        currMaxVote = currVoteCounts[index][classIndex];
                        currVoteLabels[index] = classIndex;
                    }
                }
                if (currVoteLabels[index]
                        == dset.data.get(index).getCategory()) {
                    // If correct, increase the accuracy.
                    currAccuracy++;
                }
                // No fix the neighbor occurrence frequency estimates.
                for (int kIncSecond = 0; kIncSecond < accuracyArray.length;
                        kIncSecond++) {
                    for (int j = 0; j < reverseNeighborsAllK[
                            kIncSecond][index].size(); j++) {
                        classDataKNeighborRelationAllK[kIncSecond][
                                trainingData.data.get(reverseNeighborsAllK[
                                kIncSecond][index].get(j)).getCategory()][
                                replacements[kIncSecond].get(j)]--;
                        neighbOccFreqsAllK[kMin + kIncSecond - 1][
                                replacements[kIncSecond].get(j)]--;
                    }
                }

            }
            // Normalize and update the best configuration specs.
            currAccuracy /= (float) numElements;
            accuracyArray[kInc] = currAccuracy;
            if (currMaxAcc < currAccuracy) {
                currMaxAcc = currAccuracy;
                currMaxK = kMin + kInc;
            }

        }
        k = currMaxK;
    }

    /**
     * The default constructor.
     */
    public HIKNNNonDW() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNNonDW(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNNNonDW(int k, CombinedMetric cmet) {
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
    public HIKNNNonDW(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     */
    public HIKNNNonDW(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNNNonDW(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public HIKNNNonDW(int k, float laplaceEstimator, CombinedMetric cmet,
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
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNNonDW(DataSet dset, int numClasses, CombinedMetric cmet,
            int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param nsf NeighborSetFinder object that holds and calculates kNN sets.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNNonDW(DataSet dset, int numClasses, NeighborSetFinder nsf,
            CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        this.nsf = nsf;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNNonDW(Category[] categories, CombinedMetric cmet, int k) {
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
            for (int i = 0; i < categories[cIndex].size();
                    i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].
                        getInstance(i));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
    }

    /**
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
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
        trainingData.fAttrNames =
                categories[indexFirstNonEmptyClass].getInstance(0).
                getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size();
                    i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].
                        getInstance(i));
            }
        }
        numClasses = trainingData.countCategories();
    }

    /**
     * @param trainingData DataSet object that is the training data.
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
     * @return Integer that is the neighborhood size.
     */
    public int getK() {
        return k;
    }

    /**
     * @param k Integer that is the neighborhood size.
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
     * Calculates the kNN sets.
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
        HIKNNNonDW result = new HIKNNNonDW(k, laplaceEstimator,
                getCombinedMetric(), numClasses);
        return result;
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        // Get the permutation of the indexes.
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(reducer.getPrototypeIndexes(),
                reducer.getOriginalDataSet());
        classPriors = trainingData.getClassPriors();
        int[] protoOccFreqs = reducer.getPrototypeHubness();
        neighbOccFreqs = new int[protoOccFreqs.length];
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            neighbOccFreqs[i] = protoOccFreqs[indexPermutation.get(i)];
        }
        labelInformationFactor = new float[trainingData.size()];
        float maxHubness = 0;
        float minHubness = Float.MAX_VALUE;
        // Get the maximum neighbor occcurrence frequency.
        for (int i = 0; i < trainingData.size(); i++) {
            if (neighbOccFreqs[i] > maxHubness) {
                maxHubness = neighbOccFreqs[i];
            }
            if (neighbOccFreqs[i] < minHubness) {
                minHubness = neighbOccFreqs[i];
            }
        }
        dataSize = reducer.getOriginalDataSize();
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        float[][] classDataKNeighborRelationTemp =
                reducer.getClassDataNeighborRelationforFuzzy(numClasses,
                laplaceEstimator);
        for (int c = 0; c < numClasses; c++) {
            for (int i = 0; i < neighbOccFreqs.length; i++) {
                classDataKNeighborRelation[c][i] =
                        classDataKNeighborRelationTemp[c][
                        indexPermutation.get(i)];
            }
        }
        // Calculate the label information factor.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        minEventInfo = (float) BasicMathUtil.log2(((float) dataSize)
                / (maxHubness + 1f));
        maxEventInfo = (float) BasicMathUtil.log2(((float) dataSize)
                / (0 + 1f));
        for (int i = 0; i < trainingData.size(); i++) {
            eventInfo = (float) BasicMathUtil.log2(((float) dataSize)
                    / ((float) neighbOccFreqs[i] + 1f));
            labelInformationFactor[i] = (eventInfo - minEventInfo)
                    / (maxEventInfo - minEventInfo + 0.0001f);
        }
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // In case of an invalid neighborhood size, perform automatic
            // search.
            findK(1, 20);
        }
        // If the kNN sets have not been provided, calculate them.
        if (nsf == null) {
            calculateNeighborSets();
        }
        // Calculate the class priors.
        classPriors = trainingData.getClassPriors();
        if (!noRecalc) {
            nsf.recalculateStatsForSmallerK(k);
        }
        neighbOccFreqs = nsf.getNeighborFrequencies();
        dataSize = trainingData.size();
        // Fetch the kNN sets.
        int[][] kneighbors = nsf.getKNeighbors();
        // Initialize the data structures.
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        labelInformationFactor = new float[trainingData.size()];
        int currClass;
        float maxHubness = 0;
        float minHubness = Float.MAX_VALUE;
        // Get the maximum neighbor occurrence frequency.
        for (int i = 0; i < trainingData.size(); i++) {
            if (neighbOccFreqs[i] > maxHubness) {
                maxHubness = neighbOccFreqs[i];
            }
            if (neighbOccFreqs[i] < minHubness) {
                minHubness = neighbOccFreqs[i];
            }
        }
        // Calculate the maximum and minimum neighbor occurrence informativeness
        // for normalization purposes.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        minEventInfo = (float) BasicMathUtil.log2(((float) trainingData.size())
                / (maxHubness + 1f));
        maxEventInfo = (float) BasicMathUtil.log2(((float) trainingData.size())
                / (0 + 1f));
        for (int i = 0; i < trainingData.size(); i++) {
            // Calculate the occurrence informativeness.
            eventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size()) /
                    ((float) neighbOccFreqs[i] + 1f));
            // Calculate the label information factor.
            labelInformationFactor[i] = (eventInfo - minEventInfo)
                    / (maxEventInfo - minEventInfo + 0.0001f);
            currClass = trainingData.data.get(i).getCategory();
            // Update the class-conditional occurrence frequencies.
            classDataKNeighborRelation[currClass][i]++;
            for (int j = 0; j < k; j++) {
                classDataKNeighborRelation[currClass][kneighbors[i][j]]++;
            }
        }
        // Normalize.
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < trainingData.size(); j++) {
                classDataKNeighborRelation[i][j] /= (neighbOccFreqs[j] + 1);
            }
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
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
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        // Calculate the kNN sets.
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
        float[] classProbEstimates = new float[numClasses];
        // Perform the vote.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (trainingData.data.get(kNeighbors[kIndex]).getCategory()
                        == cIndex) {
                    classProbEstimates[cIndex] +=
                            ((labelInformationFactor[kNeighbors[kIndex]]
                            + ((1 - labelInformationFactor[kNeighbors[kIndex]])
                            * classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]]))
                            * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighbOccFreqs[kNeighbors[kIndex]])));
                } else {
                    classProbEstimates[cIndex] += ((((1
                            - labelInformationFactor[kNeighbors[kIndex]]))
                            * classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]]) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighbOccFreqs[kNeighbors[kIndex]])));
                }
            }

        }
        float probTotal = 0;
        // Normalize.
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

        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (trainingData.data.get(kNeighbors[kIndex]).getCategory()
                        == cIndex) {
                    classProbEstimates[cIndex] +=
                            ((labelInformationFactor[kNeighbors[kIndex]]
                            + ((1 - labelInformationFactor[kNeighbors[kIndex]])
                            * classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]]))
                            * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighbOccFreqs[kNeighbors[kIndex]])));
                } else {
                    classProbEstimates[cIndex] += ((((1
                            - labelInformationFactor[kNeighbors[kIndex]]))
                            * classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]]) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighbOccFreqs[kNeighbors[kIndex]])));
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
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (trainingData.data.get(trNeighbors[kIndex]).getCategory()
                        == cIndex) {
                    classProbEstimates[cIndex] +=
                            ((labelInformationFactor[trNeighbors[kIndex]]
                            + ((1 - labelInformationFactor[trNeighbors[kIndex]])
                            * classDataKNeighborRelation[cIndex][
                            trNeighbors[kIndex]]))
                            * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighbOccFreqs[trNeighbors[kIndex]])));
                } else {
                    classProbEstimates[cIndex] += ((((1
                            - labelInformationFactor[trNeighbors[kIndex]]))
                            * classDataKNeighborRelation[cIndex][
                            trNeighbors[kIndex]]) * (float) BasicMathUtil.log2(
                            ((float) dataSize)
                            / (1f + neighbOccFreqs[trNeighbors[kIndex]])));
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
    public int getNeighborhoodSize() {
        return k;
    }
}
