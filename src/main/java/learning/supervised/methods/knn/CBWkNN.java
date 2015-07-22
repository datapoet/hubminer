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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;

/**
 * An algorithm described in the paper titled "Class Based Weighted K-Nearest
 * Neighbor over Imbalance Dataset" that was presented at PAKDD 2013 in Gold
 * Coast, Australia by Harshit Dubey and Vikram Pudi the idea is to assign
 * (query and class) - specific weights to instance votes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CBWkNN extends Classifier implements DistMatrixUserInterface,
        NSFUserInterface, DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    private DataSet trainingData = null;
    private int numClasses = 0;
    // Predicted labels of the training data by kNN via leave-one-out.
    private int[] knnClassifications;
    // Neighbor coefficient that is used to determine vote weights.
    private float[] neighborCoefficient;
    // Upper triangular distance matrix.
    private float[][] distMat = null;
    // Object that holds and calculates the kNN sets.
    private NeighborSetFinder nsf = null;
    // Neighborhood size.
    private int k = 10;
    // k/mValue first neighbors is used to calculate the weighting factors for
    // the query point.
    private int mValue = 2;
    
    /**
     * Default constructor.
     */
    public CBWkNN() {
    }
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("mValue", "Denominator to divide the neighborhood size by"
                + "in order to obtain the number of neighbors for weighting"
                + "calculations for the query point.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("Pacific-Asian Conference on Knowledge Discovery "
                + "and Data Mining");
        pub.addAuthor(new Author("Harshit", "Dubey"));
        pub.addAuthor(new Author("Vikram", "Pudi"));
        pub.setTitle("Class Based Weighted K-Nearest Neighbor over Imbalance "
                + "Dataset");
        pub.setYear(2013);
        pub.setStartPage(305);
        pub.setEndPage(316);
        pub.setPublisher(Publisher.SPRINGER);
        pub.setDoi("10.1007/978-3-642-37456-2_26");
        pub.setUrl("http://link.springer.com/chapter/10.1007%2F978-3-642-37456-"
                + "2_26");
        return pub;
    }

    @Override
    public void noRecalcs() {
    }

    /**
     * @param mValue Integer that is the number of neighbors to use for
     * determining the weighting factor.
     */
    public void setM(int mValue) {
        this.mValue = mValue;
    }

    /**
     * @return Integer that is the number of neighbors to use for determining
     * the weighting factor.
     */
    public int getM() {
        return mValue;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "CBWKNN";
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public CBWkNN(int numClasses, CombinedMetric cmet, int k) {
        this.numClasses = numClasses;
        if (trainingData != null) {
            knnClassifications = new int[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public CBWkNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        if (trainingData != null) {
            knnClassifications = new int[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param mValue Integer that is the number of neighbors to use for
     * determining the weighting factor.
     */
    public CBWkNN(DataSet dset, int numClasses, CombinedMetric cmet, int k,
            int mValue) {
        trainingData = dset;
        this.numClasses = numClasses;
        if (trainingData != null) {
            knnClassifications = new int[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
        this.mValue = mValue;
    }

    /**
     * Initialization.
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public CBWkNN(Category[] categories, CombinedMetric cmet, int k) {
        setClasses(categories);
        if (trainingData != null) {
            knnClassifications = new int[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new CBWkNN(trainingData, numClasses, getCombinedMetric(), k,
                mValue);
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1 &&
                    categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
                break;
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
                trainingData.addDataInstance(
                        categories[cIndex].getInstance(i));
            }
        }
        if (trainingData != null) {
            knnClassifications = new int[trainingData.size()];
        }
        numClasses = trainingData.countCategories();
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
     * @return DataSet object that is the training data.
     */
    public DataSet getTrainingSet() {
        return trainingData;
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

    /**
     * Calculate the kNN sets.
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
    public void train() throws Exception {
        // If the kNN sets have not been provided, calculate them.
        if (nsf == null) {
            calculateNeighborSets();
        }
        if (k <= 0) {
            k = nsf.getCurrK();
        }
        // Initialize the kNN classifier for leave-one-out prediction on the
        // training data.
        KNN classifier = new KNN(k, this.getCombinedMetric());
        classifier.setData(trainingData.data, trainingData);
        classifier.train();
        int labelPrediction;
        int[][] kNeighbors = null;
        float[][] kDistances = null;
        // Initialize the neighbor coefficients.
        neighborCoefficient = new float[trainingData.size()];
        if (nsf.getKNeighbors()[0].length == k) {
            kNeighbors = nsf.getKNeighbors();
            kDistances = nsf.getKDistances();
        } else if (nsf.getKNeighbors()[0].length > k) {
            kNeighbors = new int[trainingData.size()][k];
            kDistances = new float[trainingData.size()][k];
            int[][] kNeighborsLarger = nsf.getKNeighbors();
            float[][] kDistancesLarger = nsf.getKDistances();
            for (int i = 0; i < trainingData.size(); i++) {
                for (int kIndex = 0; kIndex < k; kIndex++) {
                    kNeighbors[i][kIndex] = kNeighborsLarger[i][kIndex];
                    kDistances[i][kIndex] = kDistancesLarger[i][kIndex];
                }
            }
        }
        float[] localClassCounts;
        for (int i = 0; i < trainingData.size(); i++) {
            localClassCounts = new float[numClasses];
            labelPrediction = classifier.classifyWithKDistAndNeighbors(
                    trainingData.getInstance(i), kDistances[i], kNeighbors[i]);
            knnClassifications[i] = labelPrediction;
            for (int kInd = 0; kInd < k; kInd++) {
                localClassCounts[trainingData.getLabelOf(
                        kNeighbors[i][kInd])]++;
            }
            neighborCoefficient[i] = localClassCounts[labelPrediction]
                    / (Math.max(1, localClassCounts[
                    trainingData.getLabelOf(i)]));
        }
    }

    /**
     * Calculates the class weights for the voting.
     *
     * @param neighbors int[] of neighbor indexes.
     * @return float[] of class weights based on the query neighbors.
     */
    private float[] getClassWeightsForNeighbors(int[] neighbors) {
        float[] alphas = new float[numClasses];
        float[] weights = new float[numClasses];
        int kSmall = Math.max(1, (int) ((float) k / (float) mValue));
        for (int kInd = 0; kInd < kSmall; kInd++) {
            alphas[trainingData.getLabelOf(neighbors[kInd])] +=
                    (neighborCoefficient[neighbors[kInd]] * mValue) / k;
        }
        for (int c = 0; c < numClasses; c++) {
            weights[c] = alphas[c] / (1 + alphas[c]);
        }
        return weights;
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        if (instance == null) {
            return -1;
        }
        // Calculate the kNN sets.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = cmet.dist(instance, trainingData.data.get(i));
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Make the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float probTotal = 0;
        float[] classWeights = this.getClassWeightsForNeighbors(
                nearestInstances);
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(
                    nearestInstances[kIndex]).getCategory()] += classWeights[
                    trainingData.getLabelOf(nearestInstances[kIndex])];
            probTotal += classWeights[trainingData.getLabelOf(
                    nearestInstances[kIndex])];
        }
        // Normalize.
        int maxClassIndex = -1;
        float maxProb = 0;
        if (probTotal == 0) {
            return 0;
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
                if (classProbabilities[cIndex] >= maxProb) {
                    maxClassIndex = cIndex;
                    maxProb = classProbabilities[cIndex];
                }
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        if (instance == null) {
            return null;
        }
        // Calculate the kNN sets.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = cmet.dist(instance, trainingData.data.get(i));
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float[] classWeights = this.getClassWeightsForNeighbors(
                nearestInstances);
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(
                    nearestInstances[kIndex]).getCategory()] +=
                    classWeights[trainingData.getLabelOf(nearestInstances[
                    kIndex])];
            probTotal += classWeights[trainingData.getLabelOf(
                    nearestInstances[kIndex])];
        }
        // Normalize.
        if (probTotal == 0) {
            return new float[numClasses];
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
            }
        }
        return classProbabilities;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        if (instance == null) {
            return null;
        }
        // Calculate the kNN sets.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = distToTraining[i];
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float[] classWeights = this.getClassWeightsForNeighbors(
                nearestInstances);
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(
                    nearestInstances[kIndex]).getCategory()] +=
                    classWeights[trainingData.getLabelOf(
                    nearestInstances[kIndex])];
            probTotal += classWeights[trainingData.getLabelOf(
                    nearestInstances[kIndex])];
        }
        // Normalize.
        if (probTotal == 0) {
            return new float[numClasses];
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
            }
        }
        return classProbabilities;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        if (instance == null) {
            return -1;
        }
        // Find the kNN sets.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = distToTraining[i];
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float[] classWeights = this.getClassWeightsForNeighbors(
                nearestInstances);
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(nearestInstances[kIndex]).
                    getCategory()] += classWeights[trainingData.getLabelOf(
                    nearestInstances[kIndex])];
            probTotal += classWeights[trainingData.getLabelOf(
                    nearestInstances[kIndex])];
        }
        // Normalize.
        int maxClassIndex = -1;
        float maxProb = 0;
        if (probTotal == 0) {
            return 0;
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
                if (classProbabilities[cIndex] >= maxProb) {
                    maxClassIndex = cIndex;
                    maxProb = classProbabilities[cIndex];
                }
            }
        }
        if (maxClassIndex < 0) {
            maxClassIndex = 0;
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
        if (instance == null) {
            return null;
        }
        float[] classProbabilities = new float[numClasses];
        float[] classWeights = this.getClassWeightsForNeighbors(trNeighbors);
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(trNeighbors[kIndex]).
                    getCategory()] += classWeights[trainingData.getLabelOf(
                    trNeighbors[kIndex])];
            probTotal += classWeights[trainingData.getLabelOf(
                    trNeighbors[kIndex])];
        }
        // Normalize.
        if (probTotal == 0) {
            return new float[numClasses];
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
            }
        }
        return classProbabilities;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
