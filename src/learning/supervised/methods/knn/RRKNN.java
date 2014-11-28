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
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
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
import util.AuxSort;

/**
 * This class implements the re-ranked k-nearest neighbor classifier, which
 * performs hubness-aware re-ranking of the original kNN set and then performs
 * voting by a subset of the original neighbors, the better ranked ones. The
 * ranking that is used in this algorithm was first proposed in the paper
 * Exploiting Hubs for Self-Adaptive Secondary Re-Ranking In Bug Report
 * Duplicate Detection which was presented at the ITI conference in 2013. The
 * algorithms itself was proposed in the journal paper version of the paper
 * Image Hub Explorer: Evaluating Representations and Metrics for Content-based
 * Image Retrieval and Object Recognition. Both of these papers were authored by
 * Nenad Tomasev.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RRKNN extends Classifier implements DistMatrixUserInterface,
        NSFUserInterface, DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    // The larger neighborhood size.
    private int k = 5;
    // Object for kNN calculations.
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    private float[][] distMat = null;
    // Total neighbor occurrence frequencies.
    private int[] neighbOccFreqs;
    // Bad neighbor occurrence frequencies.
    private int[] badNeighbOccFreqs;
    // The inferred distance-altering factors for re-ranking.
    private float[] multiplicativeFactors;
    // The smaller neighborhood size.
    private int kVoting;
    private DWKNN knnInternal;
    private boolean noRecalc = false;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("kVoting", "Neighborhood size to use on re-ranked sub-kNN"
                + "sets for voting and prediction.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Image Hub Explorer: Evaluating Representations and "
                + "Metrics for Content-based Image Retrieval and Object "
                + "Recognition");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.setPublisher(Publisher.ACM);
        pub.setJournalName("Multimedia Tools and Applications");
        pub.setYear(2014);
        pub.setDoi("10.1007/s11042-014-2254-1");
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "RRKNN";
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

    /**
     * The default constructor.
     */
    public RRKNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public RRKNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RRKNN(int k, CombinedMetric cmet) {
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
    public RRKNN(int k, CombinedMetric cmet, int numClasses) {
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
    public RRKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
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
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public RRKNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
    public RRKNN(Category[] categories, CombinedMetric cmet, int k) {
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
                trainingData.addDataInstance(categories[cIndex].getInstance(
                        i));
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
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
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
    public ValidateableInterface copyConfiguration() {
        RRKNN classifierCopy = new RRKNN(k, getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // If the neighborhood size wasn't specified, use the default value.
            k = 10;
        }
        // Check to see if the kNN sets were already calculated.
        if (nsf == null) {
            calculateNeighborSets();
        }
        if (!noRecalc) {
            nsf.recalculateStatsForSmallerK(k);
        }
        neighbOccFreqs = nsf.getNeighborFrequencies();
        badNeighbOccFreqs = nsf.getBadFrequencies();
        multiplicativeFactors = new float[trainingData.size()];
        // Calculate the hubness-aware multiplicative factors.
        for (int i = 0; i < trainingData.size(); i++) {
            multiplicativeFactors[i] = ((float) badNeighbOccFreqs[i])
                    / ((float) neighbOccFreqs[i] + 1);
        }
        kVoting = (int) Math.max(1, ((float) k) / 2);
        knnInternal = new DWKNN(trainingData, numClasses, getCombinedMetric(),
                kVoting);
        knnInternal.train();
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
        // Calculate the kNN set.
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
        // Update the distances based on the hubness-aware multiplicative
        // factors.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] *= multiplicativeFactors[kNeighbors[kIndex]];
        }
        // Sort, ascending.
        int[] indexPermutation = AuxSort.sortIndexedValue(kDistances, false);
        // Re-calculate the kNN set.
        float[] kDistsNew = Arrays.copyOf(kDistances, kVoting);
        int[] kNeighborsNew = new int[kVoting];
        for (int kIndex = 0; kIndex < kVoting; kIndex++) {
            kNeighborsNew[kIndex] = kNeighbors[indexPermutation[kIndex]];
        }
        return knnInternal.classifyProbabilisticallyWithKDistAndNeighbors(
                instance, kDistsNew, kNeighborsNew);
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Calculate the kNN set.
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
        // Update the distances.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] *= multiplicativeFactors[kNeighbors[kIndex]];
        }
        // Sort, ascending.
        int[] indexPermutation = AuxSort.sortIndexedValue(kDistances, false);
        // Re-calculate the kNN set.
        float[] kDistsNew = Arrays.copyOf(kDistances, kVoting);
        int[] kNeighborsNew = new int[kVoting];
        for (int i = 0; i < kVoting; i++) {
            kNeighborsNew[i] = kNeighbors[indexPermutation[i]];
        }
        return knnInternal.classifyProbabilisticallyWithKDistAndNeighbors(
                instance, kDistsNew, kNeighborsNew);
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
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining, trNeighbors);
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
        float[] kDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] = distToTraining[trNeighbors[kIndex]];
        }
        // Update the distances.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] *= multiplicativeFactors[trNeighbors[kIndex]];
        }
        // Sort, ascending.
        int[] indexPermutation = AuxSort.sortIndexedValue(kDistances, false);
        // Recalculate the kNN set.
        float[] kDistsNew = Arrays.copyOf(kDistances, kVoting);
        int[] kNeighborsNew = new int[kVoting];
        for (int kIndex = 0; kIndex < kVoting; kIndex++) {
            kNeighborsNew[kIndex] = trNeighbors[indexPermutation[kIndex]];
        }
        return knnInternal.classifyProbabilisticallyWithKDistAndNeighbors(
                instance, kDistsNew, kNeighborsNew);
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
