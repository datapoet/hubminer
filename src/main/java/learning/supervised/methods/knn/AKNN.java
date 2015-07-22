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
import data.representation.util.DataMineConstants;
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

/**
 * This class implements the algorithm proposed in the paper "Improving nearest
 * neighbor rule with a simple adaptive distance measure" published in Pattern
 * Recognition Letters (2007) by Jigang Wang, Predrag Neskovic and Leon N.
 * Cooper. We define a sphere around each element which has a radius that
 * extends all the way to the first point that does not belong to the same
 * class. This acts as a correction when calculating the distances in the point
 * of interest.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AKNN extends Classifier implements DistMatrixUserInterface,
        DistToPointsQueryUserInterface, Serializable {

    private static final long serialVersionUID = 1L;
    // The neighborhood size.
    private int k = 5;
    // The training data.
    private DataSet trainingData = null;
    // Number of classes in the data.
    private int numClasses = 0;
    // The upper triangular distance matrix.
    private float[][] distMat;
    // The prior class distribution.
    private float[] classPriors;
    // The corrective diameters for the adaptive kNN rule.
    private float[] diameters;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Improving Nearest Neighbor Rule With a Simple Adaptive "
                + "Distance Measure");
        pub.addAuthor(new Author("Jigang", "Wang"));
        pub.addAuthor(new Author("Predrag", "Neskovic"));
        pub.addAuthor(new Author("Leon N.", "Cooper"));
        pub.setPublisher(Publisher.ELSEVIER);
        pub.setJournalName("Pattern Recognition Letters");
        pub.setYear(2007);
        pub.setStartPage(207);
        pub.setEndPage(213);
        pub.setVolume(28);
        pub.setIssue(2);
        return pub;
    }

    @Override
    public String getName() {
        return "AKNN";
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
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
     * The default constructor.
     */
    public AKNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public AKNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public AKNN(int k, CombinedMetric cmet) {
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
    public AKNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
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
    public AKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public AKNN(DataSet dset, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = dset.countCategories();
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
    public AKNN(Category[] categories, CombinedMetric cmet, int k) {
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
    public ValidateableInterface copyConfiguration() {
        AKNN classifierCopy = new AKNN(k, getCombinedMetric(), numClasses);
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        // Obtain the class priors.
        classPriors = trainingData.getClassPriors();
        if (distMat == null) {
            distMat = trainingData.calculateDistMatrixMultThr(
                    getCombinedMetric(), 8);
        }
        // Initialize the corrective diameters array.
        diameters = new float[trainingData.size()];
        float minInterClassDistance;
        float currDist;
        for (int i = 0; i < trainingData.size(); i++) {
            minInterClassDistance = Float.MAX_VALUE;
            // Find the minimal inter-class distance for the query point.
            for (int j = 0; j < trainingData.size(); j++) {
                if (trainingData.getLabelOf(i) != trainingData.getLabelOf(j)) {
                    currDist = getDistanceForElements(distMat, i, j);
                    if (currDist < minInterClassDistance) {
                        minInterClassDistance = currDist;
                    }
                }
            }
            // Set the corrective diameter.
            if (!DataMineConstants.isAcceptableFloat(minInterClassDistance)) {
                diameters[i] = 0.0001f;
            } else {
                diameters[i] = minInterClassDistance
                        - DataMineConstants.EPSILON;
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
    private static float getDistanceForElements(float[][] distMatrix,
            int i, int j) {
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
        // First find the k-nearest neighbors.
        float[] kDistances = new float[k];
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDistance = cmet.dist(trainingData.data.get(i), instance);
            // Scale by the corrective diameter.
            currDistance /= diameters[i];
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
        float probTotal = 0;
        // Perform the vote.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(kNeighbors[kIndex])]++;
            probTotal++;
        }
        // Normalize the probabilities.
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
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDistance = distToTraining[i];
            // Scale by the corrective diameter.
            currDistance /= diameters[i];
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
        float probTotal = 0;
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbEstimates[trainingData.getLabelOf(kNeighbors[kIndex])]++;
            probTotal++;
        }
        // Normalize the probabilities.
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
    
}
