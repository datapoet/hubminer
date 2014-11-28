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
import learning.supervised.evaluation.ValidateableInterface;
import distances.primary.CombinedMetric;
import data.representation.DataInstance;
import data.representation.DataSet;
import learning.supervised.Category;
import learning.supervised.Classifier;
import data.neighbors.NeighborSetFinder;
import java.io.Serializable;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the basic k-nearest neighbor classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KNN extends Classifier implements AutomaticKFinderInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The training dataset.
    private DataSet trainingData = null;
    // The number of classes in the data.
    private int numClasses = 0;
    // The neighborhood size.
    private int k = 1;
    // The prior class distribution.
    private float[] classPriors;
    
    /**
     * Default constructor.
     */
    public KNN() {
    }
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Nearest Neighbor Pattern Classification");
        pub.addAuthor(new Author("T. M.", "Cover"));
        pub.addAuthor(new Author("P. E.", "Hart"));
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Information Theory");
        pub.setYear(1967);
        pub.setStartPage(21);
        pub.setEndPage(27);
        pub.setVolume(13);
        pub.setIssue(1);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "kNN";
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public KNN(int k, CombinedMetric cmet) {
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public KNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
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

    /**
     * Initialization.
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public KNN(Category[] categories, CombinedMetric cmet, int k) {
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

    @Override
    public ValidateableInterface copyConfiguration() {
        return new KNN(trainingData, numClasses, getCombinedMetric(), k);
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        numClasses = trainingData.countCategories();
        NeighborSetFinder nsf = new NeighborSetFinder(trainingData,
                getCombinedMetric());
        nsf.calculateDistances();
        nsf.calculateNeighborSets(kMax);
        // The array that holds the accuracy for the entire range of tested
        // neighborhood sizes.
        float[] accuracyArray = new float[kMax - kMin + 1];
        // The current best achieved accuracy.
        float currMaxAcc = -1f;
        // The current optimal neighborhood size.
        int currMaxK = 0;
        int dataSize = trainingData.size();
        // Votes and decisions are updated incrementally, which reduces the
        // computational complexity.
        float[][] currVoteClassCounts = new float[dataSize][numClasses];
        int[] currPredictions = new int[dataSize];
        // The label of the current vote.
        int voteLabel;
        // The k-nearest neighbor sets on the training data.
        int[][] kneighbors = nsf.getKNeighbors();
        // The current accuracy.
        float currAccuracy;
        for (int kInc = 0; kInc < accuracyArray.length; kInc++) {
            currAccuracy = 0;
            // Find the accuracy of the method on the training data for the
            // given k value.
            for (int catIndex = 0; catIndex < getClasses().length; catIndex++) {
                for (int i = 0; i < dataSize; i++) {
                    voteLabel = trainingData.getLabelOf(
                            kneighbors[i][kMin + kInc - 1]);
                    currVoteClassCounts[i][voteLabel]++;
                    if (currPredictions[i] != voteLabel) {
                        // Check if the decision needs to be updated.
                        if (currVoteClassCounts[i][voteLabel]
                                > currVoteClassCounts[i][currPredictions[i]]) {
                            currPredictions[i] = voteLabel;
                        }
                    }
                    if (currPredictions[i] == trainingData.getLabelOf(i)) {
                        currAccuracy++;
                    }
                }
            }
            // Normalize the accuracy.
            currAccuracy /= (float) dataSize;
            accuracyArray[kInc] = currAccuracy;
            // Update the best parameter values.
            if (currMaxAcc < currAccuracy) {
                currMaxAcc = currAccuracy;
                currMaxK = kMin + kInc;
            }
        }
        // Set the optimal neighborhood size as the actual one.
        k = currMaxK;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // If an invalid neighborhood size was provided, automatically
            // search for the optimal one in the lower k-range.
            findK(1, 20);
        }
        if (trainingData != null) {
            numClasses = Math.max(numClasses, trainingData.countCategories());
            classPriors = trainingData.getClassPriors();
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
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(kNeighbors[kIndex])]++;
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
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Calculate the kNN set.
        float[] kDistances = new float[k];
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            if (distToTraining[i] < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > distToTraining[i]) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = distToTraining[i];
                kNeighbors[index] = i;
            }
        }
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(kNeighbors[kIndex])]++;
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
     * Classify the point of interest based on the kNN set and the distances to
     * the neighbor points.
     *
     * @param instance DataInstance object that is to be classified.
     * @param kDistances float[] representing the distances to the k-nearest
     * neighbors.
     * @param trNeighbors int[] representing the indexes of the kNN set.
     * @return Integer that is the predicted class affiliation in the point of
     * interest.
     * @throws Exception
     */
    public int classifyWithKDistAndNeighbors(DataInstance instance,
            float[] kDistances, int[] trNeighbors) throws Exception {
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(trNeighbors[kIndex])]++;
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
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbEstimates[cIndex] > maxProb) {
                maxProb = classProbEstimates[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    /**
     * Classify the point of interest based on the kNN set and the distances to
     * the neighbor points.
     *
     * @param instance DataInstance object that is to be classified.
     * @param kDistances float[] representing the distances to the k-nearest
     * neighbors.
     * @param trNeighbors int[] representing the indexes of the kNN set.
     * @return float[] that is the predicted class distribution in the point of
     * interest.
     * @throws Exception
     */
    public float[] classifyProbabilisticallyWithKDistAndNeighbors(
            DataInstance instance, float[] kDistances, int[] trNeighbors)
            throws Exception {
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(trNeighbors[kIndex])]++;
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
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(trNeighbors[kIndex])]++;
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
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }
}