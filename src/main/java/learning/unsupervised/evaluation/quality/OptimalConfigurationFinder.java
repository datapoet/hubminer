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
package learning.unsupervised.evaluation.quality;

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;

/**
 * Helps with determining the optimal clustering of the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class OptimalConfigurationFinder {

    // The type of quality index to use.
    public static final int SILHOUETTE_INDEX = 0;
    public static final int DAVIES_BOULDIN_INDEX = 1;
    public static final int RS_INDEX = 2;
    public static final int SD_VALIDITY_INDEX = 3;
    public static final int DUNN_INDEX = 4;
    public static final int C_INDEX = 5;
    public static final int JACCARD_INDEX = 6;
    public static final int NUM_INDEXES = 7;
    // An array of cluster arrays representing different cluster configurations.
    private Cluster[][] configurations;
    DataSet dataContext;
    CombinedMetric cmet;
    private int qualityIndex = 0;

    /**
     * @param qualityIndex The quality index to use.
     */
    public void setQualityIndex(int qualityIndex) {
        this.qualityIndex = qualityIndex % NUM_INDEXES;
    }

    /**
     * @param configurations The configurations to select the best from.
     */
    public void setConfigurationList(Cluster[][] configurations) {
        this.configurations = configurations;
    }

    /**
     * @param dataset The global dataset.
     */
    public void setDataSet(DataSet dataset) {
        this.dataContext = dataset;
    }

    /**
     * @param cmet Metrics object.
     */
    public void setMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    public OptimalConfigurationFinder() {
    }

    /**
     * @param configurations The configurations to select the best from.
     * @param dataset The global dataset.
     * @param cmet Metrics object.
     * @param qualityIndex The quality index to use.
     */
    public OptimalConfigurationFinder(Cluster[][] configurations,
            DataSet dataset, CombinedMetric cmet, int qualityIndex) {
        this.cmet = cmet;
        this.qualityIndex = qualityIndex % NUM_INDEXES;
        this.configurations = configurations;
        this.dataContext = dataset;
    }

    /**
     * @param configurations The configurations to select the best from.
     * @param dataset The global dataset.
     * @param qualityIndex The quality index to use.
     */
    public OptimalConfigurationFinder(Cluster[][] configurations,
            DataSet dataset, int qualityIndex) {
        this.configurations = configurations;
        this.dataContext = dataset;
        this.qualityIndex = qualityIndex % NUM_INDEXES;
    }

    /**
     * @param qualityIndex The quality index to use.
     */
    public OptimalConfigurationFinder(int qualityIndex) {
        this.qualityIndex = qualityIndex % NUM_INDEXES;
    }

    /**
     * If the global context is not given, this piece of code tried to extract
     * the dataset information from the clusters.
     *
     * @param configurations Cluster configurations.
     * @return The corresponding embedding dataset, if specified somewhere.
     */
    private DataSet getContextInfo(Cluster[][] configurations) {
        for (Cluster[] configuration : configurations) {
            if (configuration != null) {
                for (Cluster c : configuration) {
                    if (!c.isEmpty()) {
                        return c.getDefinitionDataset();
                    }
                }
            }
        }
        return null;
    }

    /**
     * Use various different indexes and find the best configuration.
     *
     * @return The best clustering of the data, according to the index product.
     * @throws Exception
     */
    public Cluster[] findBestConfigurationByIndexProduct() throws Exception {
        if (dataContext == null) {
            if (configurations != null) {
                dataContext = getContextInfo(configurations);
            }
        }
        Cluster[] bestConfiguration = configurations[configurations.length - 1];
        float highestQuality = Float.MIN_VALUE;
        float currentQuality;

        QIndexSD validityIndexSD;
        float alpha = 1;
        for (int i = 0; i < configurations.length; i++) {
            QIndexDunn validityIndexDunn = new QIndexDunn(
                    configurations[i], dataContext, cmet);
            QIndexDaviesBouldin validityIndexDB = new QIndexDaviesBouldin(
                    configurations[i], dataContext, cmet);
            QIndexRS validityIndexRS = new QIndexRS(
                    configurations[i], dataContext);
            validityIndexSD = new QIndexSD(configurations[i], dataContext);
            validityIndexSD.setAlpha(alpha);
            QIndexSilhouette validityIndexSilhouette = new QIndexSilhouette(
                    configurations[i], dataContext, cmet);
            currentQuality = validityIndexDunn.validity()
                    * validityIndexDB.validity() * validityIndexRS.validity()
                    * validityIndexSD.validity()
                    * validityIndexSilhouette.validity();
            if (currentQuality > highestQuality) {
                highestQuality = currentQuality;
                bestConfiguration = configurations[i];
            }
        }
        return bestConfiguration;
    }

    /**
     * Create a quality index object for a given configuration.
     *
     * @param configuration Cluster configuration.
     * @return The quality index object.
     */
    private ClusteringQualityIndex initializeValidityIndex(
            Cluster[] configuration) {
        ClusteringQualityIndex validityIndex = null;
        int[] clusterAssociations = Cluster.getAssociationsForClustering(
                configuration, dataContext);
        switch (qualityIndex) {
            case SILHOUETTE_INDEX:
                validityIndex = new QIndexSilhouette(
                        configuration, dataContext, cmet);
                break;
            case DAVIES_BOULDIN_INDEX:
                validityIndex = new QIndexDaviesBouldin(
                        configuration, dataContext, cmet);
                break;
            case RS_INDEX:
                validityIndex = new QIndexRS(
                        configuration, dataContext);
                break;
            case SD_VALIDITY_INDEX:
                validityIndex = new QIndexSD(
                        configuration, dataContext);
                break;
            case DUNN_INDEX:
                validityIndex = new QIndexDunn(
                        configuration, dataContext, cmet);
                break;
            case C_INDEX:
                validityIndex = new QIndexCIndex(clusterAssociations,
                        dataContext, cmet);
                break;
            case JACCARD_INDEX:
                validityIndex = new QIndexJaccard(
                        dataContext, clusterAssociations);
                break;
        }
        return validityIndex;
    }

    /**
     * Find the best clustering of the data.
     *
     * @return The best clustering, according to the selected index.
     * @throws Exception
     */
    public Cluster[] findBestConfiguration() throws Exception {
        if (dataContext == null) {
            if (configurations != null) {
                dataContext = getContextInfo(configurations);
            }
        }
        Cluster[] bestConfiguration = configurations[configurations.length - 1];
        float highestQuality = Float.MIN_VALUE;
        float currentQuality = 1f;
        for (int i = 0; i < configurations.length; i++) {
            ClusteringQualityIndex validityIndex = initializeValidityIndex(
                    configurations[i]);
            currentQuality = validityIndex.validity();
            if (currentQuality > highestQuality) {
                highestQuality = currentQuality;
                bestConfiguration = configurations[i];
            }
        }
        return bestConfiguration;
    }
}