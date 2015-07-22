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
package learning.unsupervised.evaluation.quality.experimental;

import data.generators.util.MultiGaussianMixForClusteringTesting;
import data.generators.util.OverlappingGaussianGenerator;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.unsupervised.Cluster;
import learning.unsupervised.evaluation.quality.ClusteringQualityIndex;
import learning.unsupervised.evaluation.quality.QIndexCIndex;
import learning.unsupervised.evaluation.quality.QIndexCRootK;
import learning.unsupervised.evaluation.quality.QIndexCalinskiHarabasz;
import learning.unsupervised.evaluation.quality.QIndexDaviesBouldin;
import learning.unsupervised.evaluation.quality.QIndexDunn;
import learning.unsupervised.evaluation.quality.QIndexFolkesMallows;
import learning.unsupervised.evaluation.quality.QIndexGPlusComplement;
import learning.unsupervised.evaluation.quality.QIndexGoodmanKruskal;
import learning.unsupervised.evaluation.quality.QIndexHubertsStatistic;
import learning.unsupervised.evaluation.quality.QIndexIsolation;
import learning.unsupervised.evaluation.quality.QIndexMcClainRao;
import learning.unsupervised.evaluation.quality.QIndexPBM;
import learning.unsupervised.evaluation.quality.QIndexPointBiserial;
import learning.unsupervised.evaluation.quality.QIndexRS;
import learning.unsupervised.evaluation.quality.QIndexRand;
import learning.unsupervised.evaluation.quality.QIndexSD;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import learning.unsupervised.evaluation.quality.QIndexSimplifiedSilhouette;
import learning.unsupervised.evaluation.quality.QIndexTau;
import learning.unsupervised.methods.KMeans;
import util.CommandLineParser;

/**
 * This script evaluates the robustness of various clustering configuration
 * quality indices as the dimensionality of the data is increased. It helps with
 * estimating how the quantities change and hence how they can be interpreted.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QualityIndexEvalInHighDim {

    private static final int DEFAULT_NUM_THREADS = 8;
    private static final int DEFAULT_NEIGHBORHOOD_SIZE = 5;
    private static final int DEFAULT_CLUSTERING_REPS = 20;
    private static final int MAX_NUM_CLUST_ATTEMPTS = 30;
    // Parameters of the experiment.
    private int dim;
    private int numClusters;
    // Number of threads to use in common operations: calculating distance
    // matrices and kNN sets.
    private int numCommonThreads = DEFAULT_NUM_THREADS;
    // Neighborhood size to use for some kNN-using indexes (like isolation
    // index).
    private int k = DEFAULT_NEIGHBORHOOD_SIZE;
    // The number of times that the clustering is to be performed on each
    // dataset.
    private int numClusteringReps = DEFAULT_CLUSTERING_REPS;
    // These are for the imbalanced overlapping case.
    private int minClusterSize;
    private int maxClusterSize;
    // This is for the more balanced clearly separated clusters case.
    private int numPoints;
    private int numDatasets;
    // Generated synthetic data.
    private ArrayList<DataSet> separatedGaussians;
    private ArrayList<DataSet> overlappingGaussians;
    // Calculated index values on the data.
    private ArrayList<MultiIndexEval> indexValsSeparated;
    private ArrayList<MultiIndexEval> indexValsOverlapping;
    // Calculated index values on clustered data.
    private ArrayList<ArrayList<MultiIndexEval>> clusteredIndexValsSeparated;
    private ArrayList<ArrayList<MultiIndexEval>> clusteredIndexValsOverlapping;

    /**
     * Initialization.
     *
     * @param dim Integer that is the number of dimensions to generate the
     * synthetic data by.
     * @param numDatasets Integer that is the number of datasets to generate.
     * @param numClusters Integer that is the number of clusters to generate in
     * each dataset.
     * @param numPoints Integer that is the data size to generate in the
     * balanced separated mixtures.
     * @param minClusterSize Integer that is the minimal cluster size for the
     * imbalanced overlapping mixtures.
     * @param maxClusterSize Integer that is the maximal cluster size for the
     * imbalanced overlapping mixtures.
     */
    public QualityIndexEvalInHighDim(int dim, int numDatasets, int numClusters,
            int numPoints, int minClusterSize, int maxClusterSize) {
        this.dim = dim;
        this.numClusters = numClusters;
        this.minClusterSize = minClusterSize;
        this.maxClusterSize = maxClusterSize;
        this.numPoints = numPoints;
        this.numDatasets = numDatasets;
    }

    private MultiIndexEval getEval(DataSet dset, int[] clusterAssociations,
            float[][] dMat, NeighborSetFinder nsf) throws Exception {
        MultiIndexEval resultingEval = new MultiIndexEval();
        resultingEval.dset = dset;
        resultingEval.clusterAssociations = clusterAssociations;
        Cluster[] clusterConfiguration =
                Cluster.getConfigurationFromAssociations(
                clusterAssociations, dset);
        CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
        ArrayList<IndexValueCalculator> calcList = new ArrayList<>(20);
        // First initialize all the relevant indexes.
        QIndexSilhouette silIndex = new QIndexSilhouette(numClusters,
                clusterAssociations, dset);
        silIndex.setDistanceMatrix(dMat);
        IndexValueCalculator silCalculator =
                new IndexValueCalculator(silIndex);
        calcList.add(silCalculator);
        QIndexSimplifiedSilhouette simSilIndex =
                new QIndexSimplifiedSilhouette(numClusters,
                clusterAssociations, dset, cmet);
        IndexValueCalculator simSilCalculator =
                new IndexValueCalculator(simSilIndex);
        calcList.add(simSilCalculator);
        QIndexDunn dunnIndex =
                new QIndexDunn(clusterConfiguration, dset, cmet);
        IndexValueCalculator dunnCalculator =
                new IndexValueCalculator(dunnIndex);
        calcList.add(dunnCalculator);
        QIndexDaviesBouldin dbIndex =
                new QIndexDaviesBouldin(clusterConfiguration, dset, cmet);
        IndexValueCalculator dbCalculator =
                new IndexValueCalculator(dbIndex);
        calcList.add(dbCalculator);
        QIndexIsolation isolationIndex = new QIndexIsolation(nsf,
                clusterAssociations);
        IndexValueCalculator isolationCalculator =
                new IndexValueCalculator(isolationIndex);
        calcList.add(isolationCalculator);
        QIndexCIndex cIndex =
                new QIndexCIndex(clusterAssociations, dset, cmet);
        IndexValueCalculator cCalculator =
                new IndexValueCalculator(cIndex);
        calcList.add(cCalculator);
        QIndexCRootK cRootKIndex = new QIndexCRootK(
                clusterConfiguration, dset);
        IndexValueCalculator cRootKCalculator =
                new IndexValueCalculator(cRootKIndex);
        calcList.add(cRootKCalculator);
        QIndexCalinskiHarabasz chIndex = new QIndexCalinskiHarabasz(
                numClusters, clusterAssociations, dset);
        IndexValueCalculator chCalculator =
                new IndexValueCalculator(chIndex);
        calcList.add(chCalculator);
        QIndexFolkesMallows fmIndex = new QIndexFolkesMallows(
                numClusters, clusterAssociations, dset);
        IndexValueCalculator fmCalculator =
                new IndexValueCalculator(fmIndex);
        calcList.add(fmCalculator);
        QIndexGPlusComplement gpcIndex = new QIndexGPlusComplement(
                clusterAssociations, dset, cmet);
        gpcIndex.setDistanceMatrix(dMat);
        IndexValueCalculator gpcCalculator =
                new IndexValueCalculator(gpcIndex);
        calcList.add(gpcCalculator);
        QIndexGoodmanKruskal gkIndex = new QIndexGoodmanKruskal(
                clusterAssociations, dset, cmet);
        gkIndex.setDistanceMatrix(dMat);
        IndexValueCalculator gkCalculator =
                new IndexValueCalculator(gkIndex);
        calcList.add(gkCalculator);
        QIndexHubertsStatistic hsIndex = new QIndexHubertsStatistic(
                clusterAssociations, dset, cmet);
        hsIndex.setDistanceMatrix(dMat);
        IndexValueCalculator hsCalculator =
                new IndexValueCalculator(hsIndex);
        calcList.add(hsCalculator);
        QIndexMcClainRao mcrIndex = new QIndexMcClainRao(
                clusterAssociations, dset, cmet);
        mcrIndex.setDistanceMatrix(dMat);
        IndexValueCalculator mcrCalculator =
                new IndexValueCalculator(mcrIndex);
        calcList.add(mcrCalculator);
        QIndexPBM pbmIndex =
                new QIndexPBM(numClusters, clusterAssociations, dset);
        IndexValueCalculator pbmCalculator =
                new IndexValueCalculator(pbmIndex);
        calcList.add(pbmCalculator);
        QIndexPointBiserial pbsrIndex = new QIndexPointBiserial(
                clusterAssociations, dset, cmet);
        pbsrIndex.setDistanceMatrix(dMat);
        IndexValueCalculator pbsrCalculator =
                new IndexValueCalculator(pbsrIndex);
        calcList.add(pbsrCalculator);
        QIndexRS rsIndex = new QIndexRS(clusterConfiguration, dset);
        IndexValueCalculator rsCalculator =
                new IndexValueCalculator(rsIndex);
        calcList.add(rsCalculator);
        QIndexRand randIndex = new QIndexRand(dset, clusterAssociations);
        IndexValueCalculator randCalculator =
                new IndexValueCalculator(randIndex);
        calcList.add(randCalculator);
        QIndexSD sdIndex = new QIndexSD(clusterConfiguration, dset);
        sdIndex.setAlpha(1);
        IndexValueCalculator sdCalculator =
                new IndexValueCalculator(sdIndex);
        calcList.add(sdCalculator);
        QIndexTau tauIndex =
                new QIndexTau(clusterAssociations, dset, cmet);
        tauIndex.setDistanceMatrix(dMat);
        IndexValueCalculator tauCalculator =
                new IndexValueCalculator(tauIndex);
        calcList.add(tauCalculator);
        // Perform the evaluation calculations.
        ArrayList<Thread> threads = new ArrayList<>(calcList.size());
        for (IndexValueCalculator calc: calcList) {
            Thread t = new Thread(calc);
            threads.add(t);
            t.start();
        }
        // Wait for all the threads to finish.
        for (Thread t: threads) {
            if (t != null) {
                try {
                    t.join();
                } catch (Throwable thr) {                   
                }
            }
        }
        // Write the results to the return object.
        resultingEval.silIndexVal = silCalculator.indexValue;
        resultingEval.simSilIndexVal = simSilCalculator.indexValue;
        resultingEval.dunnIndexVal = dunnCalculator.indexValue;
        resultingEval.dbIndexVal = dbCalculator.indexValue;
        resultingEval.isolationIndexVal = isolationCalculator.indexValue;
        resultingEval.cIndexVal = cCalculator.indexValue;
        resultingEval.cRootKIndexVal = cRootKCalculator.indexValue;
        resultingEval.calinskiHarabaszIndexVal = chCalculator.indexValue;
        resultingEval.folkesMallowsIndexVal = fmCalculator.indexValue;
        resultingEval.gPlusComplementIndexVal = gpcCalculator.indexValue;
        resultingEval.goodmanKruskalIndexVal = gkCalculator.indexValue;
        resultingEval.hubertsStatIndexVal = hsCalculator.indexValue;
        resultingEval.mcClainRaoIndexVal = mcrCalculator.indexValue;
        resultingEval.pbmIndexVal = pbmCalculator.indexValue;
        resultingEval.pointBiserialIndexVal = pbsrCalculator.indexValue;
        resultingEval.rsIndexVal = rsCalculator.indexValue;
        resultingEval.randIndexVal = randCalculator.indexValue;
        resultingEval.sdIndexVal = sdCalculator.indexValue;
        resultingEval.tauIndexVal = tauCalculator.indexValue;
        return resultingEval;
    }

    /**
     * This class holds the evaluation values for multiple clustering indices on
     * a single clustering of a single dataset.
     */
    private static class MultiIndexEval {

        public DataSet dset;
        public int[] clusterAssociations;
        public float silIndexVal;
        public float simSilIndexVal;
        public float dunnIndexVal;
        public float dbIndexVal;
        public float isolationIndexVal;
        public float cIndexVal;
        public float cRootKIndexVal;
        public float calinskiHarabaszIndexVal;
        public float folkesMallowsIndexVal;
        public float gPlusComplementIndexVal;
        public float goodmanKruskalIndexVal;
        public float hubertsStatIndexVal;
        public float mcClainRaoIndexVal;
        public float pbmIndexVal;
        public float pointBiserialIndexVal;
        public float rsIndexVal;
        public float randIndexVal;
        public float sdIndexVal;
        public float tauIndexVal;
        
        /**
         * @return String that is the comma-separated header with the index 
         * names in the same order as in the toString printout. 
         */
        public static String getHeader() {
            StringBuilder bld = new StringBuilder();
            bld.append("Silhouette");
            bld.append(",");
            bld.append("Simplified Silhouette");
            bld.append(",");
            bld.append("Dunn");
            bld.append(",");
            bld.append("Davies-Bouldin");
            bld.append(",");
            bld.append("Isolation Index");
            bld.append(",");
            bld.append("C Index");
            bld.append(",");
            bld.append("C Root K Index");
            bld.append(",");
            bld.append("Calinski-Harabasz");
            bld.append(",");
            bld.append("Folkes-Mallows");
            bld.append(",");
            bld.append("G+ Complement");
            bld.append(",");
            bld.append("Goodman-Kruskal");
            bld.append(",");
            bld.append("Hubert's Statistic");
            bld.append(",");
            bld.append("McClain-Rao");
            bld.append(",");
            bld.append("PBM");
            bld.append(",");
            bld.append("Point-Biserial");
            bld.append(",");
            bld.append("RS");
            bld.append(",");
            bld.append("Rand");
            bld.append(",");
            bld.append("SD");
            bld.append(",");
            bld.append("Tau");
            return bld.toString();
        }
        
        @Override
        public String toString() {
            StringBuilder bld = new StringBuilder();
            bld.append(silIndexVal);
            bld.append(",");
            bld.append(simSilIndexVal);
            bld.append(",");
            bld.append(dunnIndexVal);
            bld.append(",");
            bld.append(dbIndexVal);
            bld.append(",");
            bld.append(isolationIndexVal);
            bld.append(",");
            bld.append(cIndexVal);
            bld.append(",");
            bld.append(cRootKIndexVal);
            bld.append(",");
            bld.append(calinskiHarabaszIndexVal);
            bld.append(",");
            bld.append(folkesMallowsIndexVal);
            bld.append(",");
            bld.append(gPlusComplementIndexVal);
            bld.append(",");
            bld.append(goodmanKruskalIndexVal);
            bld.append(",");
            bld.append(hubertsStatIndexVal);
            bld.append(",");
            bld.append(mcClainRaoIndexVal);
            bld.append(",");
            bld.append(pbmIndexVal);
            bld.append(",");
            bld.append(pointBiserialIndexVal);
            bld.append(",");
            bld.append(rsIndexVal);
            bld.append(",");
            bld.append(randIndexVal);
            bld.append(",");
            bld.append(sdIndexVal);
            bld.append(",");
            bld.append(tauIndexVal);
            return bld.toString();
        }
    }

    /**
     * This class enables multi-threaded calculations of clustering quality
     * indexes in this script.
     */
    static class IndexValueCalculator implements Runnable {

        ClusteringQualityIndex qIndex;
        float indexValue = Float.NaN;

        /**
         * Initialization.
         *
         * @param qIndex ClusteringQualityIndex to evaluate.
         */
        public IndexValueCalculator(ClusteringQualityIndex qIndex) {
            this.qIndex = qIndex;
        }

        @Override
        public void run() {
            try {
                indexValue = qIndex.validity();
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * This method generates all the synthetic datasets to use for the
     * experiment.
     */
    public void generateSyntheticData() throws Exception {
        separatedGaussians = new ArrayList<>(numDatasets);
        overlappingGaussians = new ArrayList<>(numDatasets);
        DataSet dset;
        for (int dIndex = 0; dIndex < numDatasets; dIndex++) {
            // First generate the overlapping dataset. Alpha and beta are set to
            // 0.75 and 1.5 implicitly in this call.
            dset = OverlappingGaussianGenerator.generate(
                    dim, numClusters, true, minClusterSize, maxClusterSize);
            overlappingGaussians.add(dset);
            // Now generate the more clearly separated data.
            MultiGaussianMixForClusteringTesting gen =
                    new MultiGaussianMixForClusteringTesting(
                    numClusters, dim, numPoints, false);
            dset = gen.generateRandomCollection();
            separatedGaussians.add(dset);
        }
    }

    /**
     * This method evaluates the generated data and the clustered generated data
     * according to multiple clustering quality indices.
     */
    public void evaluateIndices() throws Exception {
        DataSet dset;
        float[][] dMat;
        NeighborSetFinder nsf;
        CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
        indexValsSeparated = new ArrayList<>(numDatasets);
        indexValsOverlapping = new ArrayList<>(numDatasets);
        clusteredIndexValsSeparated = new ArrayList<>(numDatasets);
        clusteredIndexValsOverlapping = new ArrayList<>(numDatasets);
        KMeans clusterer = null;
        for (int dIndex = 0; dIndex < numDatasets; dIndex++) {
            // First the overlapping dataset.
            System.out.println("Evaluating overlapping dataset " + dIndex);
            ArrayList<MultiIndexEval> clusteredValsOverlapping =
                    new ArrayList<>(numClusteringReps);
            clusteredIndexValsOverlapping.add(clusteredValsOverlapping);
            dset = overlappingGaussians.get(dIndex);
            dMat = dset.calculateDistMatrixMultThr(cmet, numCommonThreads);
            nsf = new NeighborSetFinder(dset, dMat, cmet);
            nsf.calculateNeighborSetsMultiThr(k, numCommonThreads);
            int[] labelArray = dset.obtainLabelArray();
            MultiIndexEval eval = getEval(dset, labelArray, dMat, nsf);
            indexValsOverlapping.add(eval);
            for (int clustRep = 0; clustRep < numClusteringReps; clustRep++) {
                int attemptIndex = 0;
                boolean clDone = false;
                while (attemptIndex < MAX_NUM_CLUST_ATTEMPTS && !clDone) {
                    try {
                        clusterer = new KMeans(dset, cmet, numClusters);
                        clusterer.cluster();
                        clDone = true;
                    } catch (Exception e) {
                        attemptIndex++;
                    }
                }
                if (attemptIndex >= MAX_NUM_CLUST_ATTEMPTS) {
                    throw new Exception("Clustering error. Too many restarts.");
                } else {
                    if (clusterer != null) {
                        int[] clusterAssociations =
                                clusterer.getClusterAssociations();
                        eval = getEval(dset, clusterAssociations, dMat, nsf);
                        clusteredValsOverlapping.add(eval);
                    }
                }
            }
            
            // Then the clearly separated dataset.
            System.out.println("Evaluating separated dataset " + dIndex);
            dset = separatedGaussians.get(dIndex);
            ArrayList<MultiIndexEval> clusteredValsSeparated =
                    new ArrayList<>(numClusteringReps);
            clusteredIndexValsSeparated.add(clusteredValsSeparated);
            dMat = dset.calculateDistMatrixMultThr(cmet, numCommonThreads);
            nsf = new NeighborSetFinder(dset, dMat, cmet);
            nsf.calculateNeighborSetsMultiThr(k, numCommonThreads);
            labelArray = dset.obtainLabelArray();
            eval = getEval(dset, labelArray, dMat, nsf);
            indexValsSeparated.add(eval);
            for (int clustRep = 0; clustRep < numClusteringReps; clustRep++) {
                int attemptIndex = 0;
                boolean clDone = false;
                while (attemptIndex < MAX_NUM_CLUST_ATTEMPTS && !clDone) {
                    try {
                        clusterer = new KMeans(dset, cmet, numClusters);
                        clusterer.cluster();
                        clDone = true;
                    } catch (Exception e) {
                        attemptIndex++;
                    }
                }
                if (attemptIndex >= MAX_NUM_CLUST_ATTEMPTS) {
                    throw new Exception("Clustering error. Too many restarts.");
                } else {
                    if (clusterer != null) {
                        int[] clusterAssociations =
                                clusterer.getClusterAssociations();
                        eval = getEval(dset, clusterAssociations, dMat, nsf);
                        clusteredValsSeparated.add(eval);
                    }
                }
            }
            System.gc();
        }
    }

    /**
     * This method prints out the results.
     * 
     * @param outFile File to print the results to.
     * @throws IOException 
     */
    public void printResults(File outFile) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            for (int dIndex = 0; dIndex < numDatasets; dIndex++) {
                MultiIndexEval eval = indexValsSeparated.get(dIndex);
                pw.println("dataset_separated_" + dIndex);
                pw.println("ground_truth_eval");
                pw.println(MultiIndexEval.getHeader());
                pw.println(eval);
                pw.println("clustering_evals");
                pw.println(MultiIndexEval.getHeader());
                ArrayList<MultiIndexEval> clusteredValsSeparated =
                        clusteredIndexValsSeparated.get(dIndex);
                for (MultiIndexEval evalClust: clusteredValsSeparated) {
                    pw.println(evalClust);
                }
                pw.println();
            }
            for (int dIndex = 0; dIndex < numDatasets; dIndex++) {
                MultiIndexEval eval = indexValsOverlapping.get(dIndex);
                pw.println("dataset_overlapping_" + dIndex);
                pw.println("ground_truth_eval");
                pw.println(MultiIndexEval.getHeader());
                pw.println(eval);
                pw.println("clustering_evals");
                pw.println(MultiIndexEval.getHeader());
                ArrayList<MultiIndexEval> clusteredValsOverlapping =
                        clusteredIndexValsOverlapping.get(dIndex);
                for (MultiIndexEval evalClust: clusteredValsOverlapping) {
                    pw.println(evalClust);
                }
                pw.println();
            }
        } catch (IOException e) {
            throw e;
        }
    }

    /**
     * @param numCommonThreads Integer value that is the number of common
     * threads to use for operations like calculating the distance matrix and
     * the kNN sets.
     */
    public void setNumCommonThreads(int numCommonThreads) {
        this.numCommonThreads = numCommonThreads;
    }

    /**
     * @param k Integer value that is the neighborhood size to use for indexes
     * like the isolation index where the kNN sets are required.
     */
    public void setNeighborhoodSize(int k) {
        this.k = k;
    }

    /**
     * @param numClusteringReps Integer value that is the number of times that
     * the clustering is to be performed on each dataset.
     */
    public void setNumClusteringReps(int numClusteringReps) {
        this.numClusteringReps = numClusteringReps;
    }

    /**
     * This script runs the experiments to evaluate various clustering quality
     * indices on synthetic data of the specified dimensionality.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-outFile", "Output path.", CommandLineParser.STRING,
                true, false);
        clp.addParam("-numPoints", "The number of points to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-minClusterSize", "Minimal cluster size to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-maxClusterSize", "Maximal cluster size to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numClusters", "Number of clusters to generate in each "
                + "dataset.", CommandLineParser.INTEGER, true, false);
        clp.addParam("-numDatasets", "Number of datasets to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-dim", "Dimensionality of the data.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-k", "Neighborhood size for kNN-using indexes.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numCommonThreads", "Dimensionality of the data.",
                CommandLineParser.INTEGER, false, false);
        clp.addParam("-clusteringRepsPerDataset", "How many times to perform "
                + "clustering on each dataset.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File outFile = new File((String) (clp.getParamValues(
                "-outFile").get(0)));
        int minClusterSize = (Integer) (
                clp.getParamValues("-minClusterSize").get(0));
        int maxClusterSize = (Integer) (
                clp.getParamValues("-maxClusterSize").get(0));
        int numPoints = (Integer) (clp.getParamValues("-numPoints").get(0));
        int numClusters = (Integer) (clp.getParamValues("-numClusters").get(0));
        int numDatasets = (Integer) (clp.getParamValues("-numDatasets").get(0));
        int dim = (Integer) (clp.getParamValues("-dim").get(0));
        int k = (Integer) (clp.getParamValues("-k").get(0));
        int numCommonThreads = (Integer) (
                clp.getParamValues("-numCommonThreads").get(0));
        int clusteringRepsPerDataset = (Integer) (
                clp.getParamValues("-clusteringRepsPerDataset").get(0));
        QualityIndexEvalInHighDim evaluator = new QualityIndexEvalInHighDim(
                dim, numDatasets, numClusters, numPoints,
                minClusterSize, maxClusterSize);
        evaluator.setNumCommonThreads(numCommonThreads);
        evaluator.setNeighborhoodSize(k);
        evaluator.setNumClusteringReps(clusteringRepsPerDataset);
        evaluator.generateSyntheticData();
        evaluator.evaluateIndices();
        evaluator.printResults(outFile);
    }
}
