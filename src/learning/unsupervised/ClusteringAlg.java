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
package learning.unsupervised;

import algref.Citable;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;

import java.io.File;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the functionality for a clustering algorithm.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class ClusteringAlg extends Thread implements Citable {

    // The minimal number of iterations to run in iterative clustering methods.
    public static final int MIN_ITERATIONS = 15;
    // The maximal number of retries after clustering exceptions are thrown. In
    // some methods, like k-means, an empty cluster may occur during the
    // iterations, which causes exceptions to be thrown. This can be the case
    // when the number of clusters is set to a very high value. Such methods
    // can try to re-cluster the data with a different seed multiple times and
    // this constant sets the upper bound before giving up.
    public static final int MAX_RETRIES = 10;
    private DataSet dset;
    private int numClusters = 1;
    // Instance-to-cluster associations.
    private int[] clusterAssociations;
    // A flag indicating whether the algorithm is currently running.
    private volatile boolean running = false;
    // The iteration counter.
    private int iteration = 0;
    // The object responsible for distance calculations.
    private CombinedMetric cmet = null;
    // The log file.
    private File clusteringLog;
    
    /**
     * @return HashMap<String, String> that maps the parameters used in the
     * algorithm to their descriptions.
     */
    public abstract HashMap<String, String> getParameterNamesAndDescriptions();

    /**
     * @param dset DataSet to be clustered.
     */
    public void setDataSet(DataSet dset) {
        this.dset = dset;
    }

    /**
     * @return DataSet that is being clustered.
     */
    public DataSet getDataSet() {
        return dset;
    }

    /**
     * Set the 'running' flag to true to indicate the clusterer is busy.
     */
    public void flagAsActive() {
        running = true;
    }

    /**
     * Indicate the clusterer has finished the analysis.
     */
    public void flagAsInactive() {
        running = false;
    }

    /**
     * @return True if the algorithm is running, false otherwise.
     */
    public boolean isActive() {
        return running;
    }

    /**
     * Specify logging details.
     *
     * @param logFile File where the log will be written.
     */
    public void setLogTarget(File logFile) {
        this.clusteringLog = logFile;
    }

    /**
     * Learn logging details.
     *
     * @return logFile File where the log will be written.
     */
    public File getLogTarget() {
        return clusteringLog;
    }

    /**
     * @return Array defining associations between instances and clusters.
     */
    public int[] getClusterAssociations() {
        return clusterAssociations;
    }

    /**
     * Set the association of a particular instance.
     *
     * @param index Index of the data instance.
     * @param clusterNumber Cluster to set its association to.
     */
    public void setClusterAssociationsOf(int index, int clusterNumber) {
        clusterAssociations[index] = clusterNumber;
    }

    /**
     * Sets the integer cluster associations array.
     *
     * @param clusterAssociations Array defining associations between instances
     * and clusters.
     */
    public void setClusterAssociations(int[] clusterAssociations) {
        this.clusterAssociations = clusterAssociations;
    }

    /**
     * @param resetDefault The value to reset the array to.
     * @return Array defining associations between instances and clusters.
     */
    public void resetClusterAssociations(int resetDefault) {
        if (clusterAssociations == null) {
            return;
        }
        Arrays.fill(clusterAssociations, 0, clusterAssociations.length,
                resetDefault);
    }

    /**
     * @return Number of clusters.
     */
    public int getNumClusters() {
        return numClusters;
    }

    /**
     * @param numClusters Number of clusters to cluster to.
     */
    public void setNumClusters(int numClusters) {
        this.numClusters = numClusters;
    }

    /**
     * Sets the index of current iteration.
     *
     * @param iteration
     */
    public void setIterationIndex(int iteration) {
        this.iteration = iteration;
    }

    /**
     * @return Index of the current iteration.
     */
    public int getIterationIndex() {
        return iteration;
    }

    /**
     * Increases the iteration counter.
     */
    public void nextIteration() {
        iteration++;
    }

    /**
     * Sets the minimum number of iterations to run.
     *
     * @param minIterations
     */
    public void setMinIterations(int minIterations) {
        // Meant to be overridden, if and when applicable.
    }

    /**
     * Sets the combined metric used for distance calculations.
     *
     * @param cmet The metrics object.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * Gets the combined metric used for distance calculations.
     *
     * @return The metrics object.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    @Override
    public void run() {
        try {
            cluster();
        } catch (Exception e) {
            PrintWriter logWriter = null;
            try {
                if ((clusteringLog != null) && clusteringLog.exists()
                        && clusteringLog.isFile()) {
                    logWriter = new PrintWriter(new FileWriter(clusteringLog));
                    logWriter.println(e);
                    System.err.println(e);
                }
            } catch (Exception eInternal) {
                System.err.println(eInternal);
            } finally {
                if (logWriter != null) {
                    logWriter.close();
                }
            }
        }
    }

    /**
     * The method that actually does the clustering.
     *
     * @throws Exception
     */
    public abstract void cluster() throws Exception;

    /**
     * Given the clustering model, determine assignments for newly observed
     * points.
     *
     * @param dset Dataset.
     * @param nsfTest NeighborSetFinder object for the test data. Not always
     * necessary.
     * @return
     */
    public abstract int[] assignPointsToModelClusters(DataSet dset,
            NeighborSetFinder nsfTest);

    /**
     * Given the clustering model, determine assignments for newly observed
     * points.
     *
     * @param dset Dataset.
     * @param nsfTest NeighborSetFinder object for the test data. Not always
     * necessary.
     * @param kernelSimToTraining Kernel similarity array to training data.
     * @param selfKernels Kernel self-products.
     * @return
     */
    public int[] assignPointsToModelClusters(DataSet dset,
            NeighborSetFinder nsfTest, float[][] kernelSimToTraining,
            float[] selfKernels) {
        return assignPointsToModelClusters(dset, nsfTest);
    }

    /**
     * Gets the clusters for the current cluster association array.
     *
     * @return
     */
    public Cluster[] getClusters() {
        return Cluster.getConfigurationFromAssociations(clusterAssociations,
                dset);
    }

    /**
     * Gets the configuration that had a minimum error function, instead of
     * returning the final one. In most methods, the two are equivalent - but
     * this does not need to be the case in general.
     *
     * @return
     */
    public Cluster[] getMinimizingClusters() {
        return getClusters(); // Overridden in some classes.
    }

    /**
     * This method performs the basic checks before the clustering starts.
     *
     * @throws Exception
     */
    public void performBasicChecks() throws Exception {
        if (dset == null || dset.isEmpty()) {
            throw new Exception("No data provided for clustering.");
        }
        if (!DataMineConstants.isAcceptableInt(numClusters) || numClusters <= 0
                || numClusters > dset.size()) {
            throw new Exception("Inappropriate cluster number: " + numClusters);
        }
    }

    /**
     * Handles the case of trivial cluster assignments - when all points belong
     * to a single cluster or each point belongs to its own.
     *
     * @return
     */
    public boolean checkIfTrivial() {
        if (dset == null) {
            return true;
        }
        if (numClusters == 1) {
            int[] clustAssociations = new int[dset.size()];
            setClusterAssociations(clustAssociations);
            return true;
        }
        if (numClusters == dset.size()) {
            int[] clustAssociations = new int[dset.size()];
            for (int i = 0; i < clustAssociations.length; i++) {
                clustAssociations[i] = i;
            }
            setClusterAssociations(clustAssociations);
            return true;
        }
        return false;
    }
}