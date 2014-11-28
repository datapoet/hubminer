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
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;

/**
 * Implements the algorithm described in Neighbor-weighted K-nearest neighbor
 * for unbalanced text corpus by Songbo Tan in Expert Systems with Applications
 * 28 (2005) 667–671. A weighting factor is included in the voting procedure to
 * compensate for class imbalance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NWKNN extends Classifier implements DistMatrixUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private int k = 5;
    private DataSet trainingData = null;
    private int numClasses = 0;
    private float[][] distMat;
    private float[] classPriors;
    private float[] classWeights;
    private float weightExponent = 0.25f;
    private float mValue = 2;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("weightExponent", "Exponent for class-specific vote "
                + "weights.");
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Neighbor-weighted K-nearest neighbor for unbalanced text"
                + " corpus");
        pub.addAuthor(new Author("Songbo", "Tan"));
        pub.setPublisher(Publisher.ELSEVIER);
        pub.setJournalName("Expert Systems with Applications");
        pub.setYear(2005);
        pub.setStartPage(667);
        pub.setEndPage(671);
        pub.setVolume(28);
        pub.setIssue(4);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "NWKNN";
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
     * @return Float value that is the weight exponent.
     */
    public float getWeightExponent() {
        return weightExponent;
    }

    /**
     * @param exponent Float value that is the weight exponent.
     */
    public void setWeightExponent(float exponent) {
        this.weightExponent = exponent;
    }

    /**
     * Default constructor.
     */
    public NWKNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public NWKNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NWKNN(int k, CombinedMetric cmet) {
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
    public NWKNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     * @param exponent Float value that is the weight exponent.
     */
    public NWKNN(int k, CombinedMetric cmet, int numClasses, float exponent) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
        this.weightExponent = exponent;
    }

    /**
     * Initialization.
     *
     * @param dset DataSrt object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public NWKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSrt object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param exponent Float value that is the weight exponent.
     */
    public NWKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k,
            float exponent) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
        this.weightExponent = exponent;
    }

    /**
     * Initialization.
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public NWKNN(Category[] categories, CombinedMetric cmet, int k) {
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
        for (int cFirst = 0; cFirst < categories.length; cFirst++) {
            for (int cSecond = 0; cSecond < categories[cFirst].size();
                    cSecond++) {
                categories[cFirst].getInstance(cSecond).setCategory(cFirst);
                trainingData.addDataInstance(categories[cFirst].getInstance(
                        cSecond));
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
            if (indexFirstNonEmptyClass == -1 &&
                    categories[cIndex].size() > 0) {
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
        for (int cFirst = 0; cFirst < categories.length; cFirst++) {
            for (int cSecond = 0; cSecond < categories[cFirst].size();
                    cSecond++) {
                categories[cFirst].getInstance(cSecond).setCategory(cFirst);
                trainingData.addDataInstance(categories[cFirst].getInstance(
                        cSecond));
            }
        }
        numClasses = trainingData.countCategories();
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

    @Override
    public ValidateableInterface copyConfiguration() {
        NWKNN result = new NWKNN(k, getCombinedMetric(), numClasses);
        return result;
    }

    @Override
    public void train() throws Exception {
        classPriors = trainingData.getClassPriors();
        classWeights = new float[classPriors.length];
        float minSize = Float.MAX_VALUE;
        // Find the class of minimum size.
        for (int cIndex = 0; cIndex < classWeights.length; cIndex++) {
            if (classPriors[cIndex] < minSize && classPriors[cIndex] > 0) {
                minSize = classPriors[cIndex];
            }
        }
        // Set the class weights.
        for (int cIndex = 0; cIndex < classWeights.length; cIndex++) {
            if (classPriors[cIndex] > 0) {
                classWeights[cIndex] = 1f
                        / (float) Math.pow(classPriors[cIndex]
                        / minSize, weightExponent);
            } else {
                classWeights[cIndex] = 0;
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
        // Calculate the kNN sets.
        float[] kDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] = Float.MAX_VALUE;
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
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(kNeighbors[kIndex])] +=
                    (classWeights[trainingData.getLabelOf(kNeighbors[kIndex])]
                    * distanceWeights[kIndex] / dwSum);
        }
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
        // Calculate the kNN sets.
        float[] kDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] = Float.MAX_VALUE;
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
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(kNeighbors[kIndex])] +=
                    (classWeights[trainingData.getLabelOf(kNeighbors[kIndex])]
                    * distanceWeights[kIndex] / dwSum);
        }
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
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int i = 0; i < numClasses; i++) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i];
                maxClassIndex = i;
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
        for (int i = 0; i < numClasses; i++) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i];
                maxClassIndex = i;
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining, int[] trNeighbors) throws Exception {

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
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (trainingData.getLabelOf(trNeighbors[kIndex]) >= numClasses) {
                continue;
            }
            try {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])] += (
                        classWeights[trainingData.getLabelOf(
                        trNeighbors[kIndex])] * distanceWeights[kIndex]
                        / dwSum);
            } catch (Exception e) {
                continue;
            }
        }
        // Normalization.
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
